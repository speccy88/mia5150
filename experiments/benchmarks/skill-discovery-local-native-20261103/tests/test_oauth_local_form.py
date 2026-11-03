"""Serialized HTTP regression tests; no live browser, account, or OAuth network requests."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oauth_chatgpt as oauth

TEST_TEMP = Path(__file__).resolve().parents[1]/'.test-tmp'
TEST_TEMP.mkdir(exist_ok=True)
PORT = 1456
ORIGIN = f'http://127.0.0.1:{PORT}'
REVIEW = '/review/synthetic-only'
CSRF = 'synthetic-only'


class FakeSocket:
    def __init__(self, request):
        self.request = request
        self.response = bytearray()
    def makefile(self, *_args):
        return io.BytesIO(self.request)
    def sendall(self, data):
        self.response.extend(data)


def exchange(handler, server, method='GET', path=REVIEW, host=f'127.0.0.1:{PORT}',
             origin=None, csrf=CSRF):
    body = urlencode({'csrf': csrf}).encode() if method == 'POST' else b''
    lines = [f'{method} {path} HTTP/1.0', f'Host: {host}']
    if origin is not None:
        lines.append(f'Origin: {origin}')
    if method == 'POST':
        lines.extend(['Content-Type: application/x-www-form-urlencoded', f'Content-Length: {len(body)}'])
    sock = FakeSocket(('\r\n'.join(lines)+'\r\n\r\n').encode()+body)
    handler(sock, ('127.0.0.1', 54321), server)
    head, content = bytes(sock.response).split(b'\r\n\r\n', 1)
    fields = head.decode().split('\r\n')
    headers = dict(line.split(': ', 1) for line in fields[1:] if ': ' in line)
    return int(fields[0].split()[1]), headers, content.decode()


def browser_origin_for_same_origin_form(policy):
    # Fetch Standard, append-a-request-Origin-header: navigate-mode POST +
    # no-referrer serializes Origin as null; same-origin preserves a local origin.
    return 'null' if policy == 'no-referrer' else ORIGIN


class LocalFormTests(unittest.TestCase):
    def inside_synthetic_listener(self, scenario):
        class Server:
            def __init__(self, address, handler):
                self.handler = handler
                self.timeout = None
                self.assert_address = address
            def handle_request(self):
                scenario(self.handler, self)
                clock.return_value = 100000  # Finish the isolated listener after this scenario.
            def server_close(self):
                pass
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as private, \
             patch.object(oauth, 'PRIVATE', Path(private)), \
             patch.object(oauth, 'HTTPServer', Server), \
             patch.object(oauth.time, 'time', return_value=0) as clock, \
             patch.object(oauth.BaseHTTPRequestHandler, 'date_time_string', return_value='Tue, 03 Nov 2026 00:00:00 GMT'), \
             patch.object(oauth.secrets, 'token_urlsafe', return_value='synthetic-only'), \
             patch.object(oauth, 'atomic_bytes') as write, \
             patch.object(oauth, 'http_json') as external, \
             patch.object(oauth, 'save_profile') as save, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(oauth.OAuthError):
                oauth.login(PORT, open_browser=False)
            external.assert_not_called()
            save.assert_not_called()
            self.assertFalse((Path(private)/'host.json').exists())
            self.assertFalse((Path(private)/'connection.dpapi').exists())
            return write.call_count

    def test_browser_form_preserves_origin_and_can_reach_authorization_redirect(self):
        def scenario(handler, server):
            code, headers, body = exchange(handler, server)
            self.assertEqual(code, 200)
            self.assertIn(CSRF, body)
            origin = browser_origin_for_same_origin_form(headers['Referrer-Policy'])
            code, redirect, body = exchange(handler, server, method='POST', origin=origin)
            self.assertEqual(code, 303, body)  # Before fix: 400, Origine refusée.
            self.assertTrue(redirect['Location'].startswith(oauth.AUTHORIZE+'?'))
            self.assertEqual(redirect['Referrer-Policy'], 'no-referrer')
        self.assertEqual(self.inside_synthetic_listener(scenario), 1)  # Mocked host-ID write only.

    def test_foreign_null_missing_origin_and_bad_host_stay_rejected(self):
        def scenario(handler, server):
            for origin in ('null', None, '', 'https://example.invalid', 'http://localhost:1456',
                           'http://127.0.0.1:1455', ORIGIN+'/', ORIGIN+'.example.invalid'):
                code, _, _ = exchange(handler, server, method='POST', origin=origin)
                self.assertEqual(code, 400, f'Unexpected origin accepted: {origin!r}')
            code, _, _ = exchange(handler, server, method='POST', host='localhost:1456', origin=ORIGIN)
            self.assertEqual(code, 400)
            code, _, _ = exchange(handler, server, host='attacker.invalid:1456')
            self.assertEqual(code, 400)
        self.assertEqual(self.inside_synthetic_listener(scenario), 0)

    def test_csrf_and_callback_referrer_protection_remain(self):
        def scenario(handler, server):
            code, headers, _ = exchange(handler, server)
            self.assertEqual(headers['Referrer-Policy'], 'same-origin')
            self.assertIn("form-action 'self'", headers['Content-Security-Policy'])
            code, _, body = exchange(handler, server, method='POST', origin=ORIGIN, csrf='wrong-synthetic-value')
            self.assertEqual(code, 400)
            self.assertIn('Confirmation', body)
            code, headers, _ = exchange(handler, server, path='/auth/callback?code=SYNTHETIC&state=SYNTHETIC')
            self.assertEqual(code, 400)
            self.assertEqual(headers['Referrer-Policy'], 'no-referrer')
        self.assertEqual(self.inside_synthetic_listener(scenario), 0)

    def test_repeat_click_reuses_the_same_approved_attempt(self):
        def scenario(handler, server):
            code, first, _ = exchange(handler, server, method='POST', origin=ORIGIN)
            self.assertEqual(code, 303)
            code, _, repeated = exchange(handler, server, method='POST', origin=ORIGIN)
            self.assertEqual(code, 200)
            self.assertIn('Connexion déjà démarrée', repeated)
            import html
            self.assertIn(first['Location'], html.unescape(repeated))
            code, _, _ = exchange(handler, server, method='POST', origin='null')
            self.assertEqual(code, 400)
            code, _, _ = exchange(handler, server, method='POST', origin=ORIGIN, csrf='wrong')
            self.assertEqual(code, 400)
        self.assertEqual(self.inside_synthetic_listener(scenario), 1)

    def test_review_wait_is_two_hours_without_changing_official_oauth_expiry(self):
        self.assertEqual(oauth.REVIEW_WAIT_SECONDS, 7200)
        self.assertEqual(oauth.AUTHORIZATION_WAIT_SECONDS, 1800)
        def scenario(handler, server):
            code, headers, _ = exchange(handler, server)
            self.assertEqual(code, 200)
            self.assertEqual(headers['Referrer-Policy'], 'same-origin')
        self.assertEqual(self.inside_synthetic_listener(scenario), 0)


if __name__ == '__main__':
    unittest.main()
