"""Offline regression tests; all HTTP traffic and waits are mocked."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import urllib.error

common = types.ModuleType('installer_common')
common.digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
common.event = Mock()
common.require_time = Mock()
sys.modules['installer_common'] = common
spec = importlib.util.spec_from_file_location('fno_installer_test', Path(__file__).with_name('install_fno_phase4.py'))
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class Response:
    def __init__(self, chunks, status=206, headers=None):
        self.status = status
        self.headers = headers or {}
        self.chunks = list(chunks)

    def read(self, count):
        item = self.chunks.pop(0) if self.chunks else b''
        if isinstance(item, BaseException):
            raise item
        return item

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fno-download-test-')
        self.addCleanup(self.tmp.cleanup)
        installer.WHEELS = Path(self.tmp.name)
        self.data = b'abcdefghij'
        self.artifact = dict(name='example', filename='example.whl', size=10,
                             sha256=hashlib.sha256(self.data).hexdigest(), url='https://example.invalid/wheel')
        self.target = installer.WHEELS / 'example.whl'
        self.partial = installer.WHEELS / 'example.whl.part'
        common.event.reset_mock()
        common.require_time.reset_mock(side_effect=True)
        self.network = patch.object(installer.urllib.request, 'urlopen').start()
        self.pause = patch.object(installer.time, 'sleep').start()
        self.addCleanup(patch.stopall)

    def response(self, offset=0, chunks=None):
        return Response(chunks if chunks is not None else [self.data[offset:], b''], headers={
            'Content-Range': f'bytes {offset}-9/10', 'Content-Length': str(10-offset)})

    def test_complete_wheel_reused_without_network(self):
        self.target.write_bytes(self.data)
        installer.download(self.artifact)
        self.network.assert_not_called()

    def test_invalid_complete_wheel_not_overwritten(self):
        self.target.write_bytes(b'bad')
        with self.assertRaises(RuntimeError): installer.download(self.artifact)
        self.assertEqual(self.target.read_bytes(), b'bad')
        self.network.assert_not_called()

    def test_existing_partial_resumed(self):
        self.partial.write_bytes(self.data[:4])
        self.network.return_value = self.response(4)
        installer.download(self.artifact)
        self.assertEqual(self.network.call_args.args[0].get_header('Range'), 'bytes=4-')
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_read_timeout_resumes_written_offset(self):
        self.network.side_effect = [self.response(chunks=[self.data[:4], TimeoutError('read timeout')]), self.response(4)]
        installer.download(self.artifact)
        self.assertEqual(self.network.call_args_list[1].args[0].get_header('Range'), 'bytes=4-')
        self.assertEqual(self.target.read_bytes(), self.data)
        self.pause.assert_called_once_with(5)

    def test_connect_timeout_retries(self):
        self.network.side_effect = [TimeoutError('connect timeout'), self.response()]
        installer.download(self.artifact)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(self.network.call_args.kwargs['timeout'], 90)

    def test_range_ignored_restarts_only_partial(self):
        self.partial.write_bytes(b'abcd')
        self.network.return_value = Response([self.data, b''], status=200, headers={'Content-Length': '10'})
        installer.download(self.artifact)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_wrong_range_preserves_partial(self):
        self.partial.write_bytes(b'abcd')
        self.network.return_value = self.response(3)
        with self.assertRaises(RuntimeError): installer.download(self.artifact)
        self.assertEqual(self.partial.read_bytes(), b'abcd')
        self.pause.assert_not_called()

    def test_full_response_bad_length_preserves_partial(self):
        self.partial.write_bytes(b'abcd')
        self.network.return_value = Response([b'bad'], status=200, headers={'Content-Length': '3'})
        with self.assertRaises(RuntimeError): installer.download(self.artifact)
        self.assertEqual(self.partial.read_bytes(), b'abcd')

    def test_early_eof_resumes(self):
        self.network.side_effect = [self.response(chunks=[b'abcd', b'']), self.response(4)]
        installer.download(self.artifact)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_retry_limit_preserves_partial(self):
        self.partial.write_bytes(b'abcd')
        self.network.side_effect = TimeoutError('unavailable')
        with self.assertRaisesRegex(RuntimeError, 'retry limit'): installer.download(self.artifact)
        self.assertEqual(self.network.call_count, 5)
        self.assertEqual([c.args[0] for c in self.pause.call_args_list], [5, 10, 20, 40])
        self.assertEqual(self.partial.read_bytes(), b'abcd')

    def test_sha256_mismatch_stops(self):
        self.network.return_value = self.response(chunks=[b'XXXXXXXXXX', b''])
        with self.assertRaisesRegex(RuntimeError, 'SHA256'): installer.download(self.artifact)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.partial.exists())

    def test_disk_error_not_retried(self):
        self.network.return_value = self.response()
        with patch.object(installer.os, 'fsync', side_effect=OSError('disk error')):
            with self.assertRaises(OSError): installer.download(self.artifact)
        self.assertEqual(self.network.call_count, 1)
        self.pause.assert_not_called()

    def test_deadline_stops_before_network(self):
        common.require_time.side_effect = RuntimeError('Insufficient time')
        with self.assertRaises(RuntimeError): installer.download(self.artifact)
        self.network.assert_not_called()

    def test_http_403_not_retried(self):
        self.network.side_effect = urllib.error.HTTPError(self.artifact['url'], 403, 'Forbidden', {}, io.BytesIO())
        with self.assertRaises(urllib.error.HTTPError): installer.download(self.artifact)
        self.assertEqual(self.network.call_count, 1)

    def test_http_503_retried(self):
        self.network.side_effect = [urllib.error.HTTPError(self.artifact['url'], 503, 'Unavailable', {}, io.BytesIO()), self.response()]
        installer.download(self.artifact)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_complete_partial_verified_without_network(self):
        self.partial.write_bytes(self.data)
        installer.download(self.artifact)
        self.network.assert_not_called()
        self.assertEqual(self.target.read_bytes(), self.data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
