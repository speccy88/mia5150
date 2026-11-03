"""Official Sign in with ChatGPT; own registration, Windows DPAPI, no borrowed credentials.

login starts a local review page only. The user must click Continue with ChatGPT
in the Windows browser before OAuth begins. No inference is performed by login.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
from ctypes import wintypes
import hashlib
import hmac
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT/"private"
ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
APP_NAME = "Skill Discovery Benchmark"
REVIEW_WAIT_SECONDS = 2 * 60 * 60
AUTHORIZATION_WAIT_SECONDS = 30 * 60  # Client wait only; never overrides OpenAI token/code expiry.


class OAuthError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward bearer credentials or authorization codes to another target.
        raise OAuthError("Unexpected HTTP redirect; request stopped")


def http_json(url, data=None, token=None, form=False):
    headers = {}
    if data is not None:
        payload = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
    else:
        payload = None
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        # Do not print error bodies; a provider could echo codes/tokens in them.
        raise OAuthError(f"OAuth HTTP {exc.code}; no credentials printed") from None
    except (OSError, urllib.error.URLError, TimeoutError):
        raise OAuthError("OAuth network request failed") from None


class DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def dpapi(data, protect):
    if os.name != "nt":
        raise OAuthError("Credential storage requires Windows DPAPI CurrentUser")
    buf = ctypes.create_string_buffer(data)
    inp = DataBlob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    out = DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if protect:
        ok = crypt32.CryptProtectData(ctypes.byref(inp), "Skill Discovery Benchmark", None,
                                     None, None, 1, ctypes.byref(out))
    else:
        ok = crypt32.CryptUnprotectData(ctypes.byref(inp), None, None,
                                       None, None, 1, ctypes.byref(out))
    if not ok:
        raise OAuthError("Windows DPAPI failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def atomic_bytes(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_profile():
    path = PRIVATE/"connection.dpapi"
    if not path.exists():
        raise OAuthError("Not connected; run Connect-ChatGPT.ps1 on Windows")
    return json.loads(dpapi(path.read_bytes(), False))


def save_profile(profile):
    PRIVATE.mkdir(exist_ok=True)
    atomic_bytes(PRIVATE/"connection.dpapi", dpapi(json.dumps(profile).encode(), True))


def validate_id_token(token, client_id, nonce=None, expected_subject=None):
    import jwt
    config = http_json(ISSUER + "/.well-known/openid-configuration")
    jwks_uri = config.get("jwks_uri", "")
    uri = urllib.parse.urlparse(jwks_uri)
    if uri.scheme != "https" or uri.hostname != "auth.openai.com":
        raise OAuthError("Unexpected JWKS origin")
    jwks = http_json(jwks_uri)
    header = jwt.get_unverified_header(token)
    if header.get("alg") != "RS256":
        raise OAuthError("Unsupported ID token signature algorithm")
    keys = [k for k in jwks["keys"] if k.get("kid") == header.get("kid")]
    if len(keys) != 1:
        raise OAuthError("ID token signing key is unavailable")
    key = jwt.PyJWK.from_dict(keys[0]).key
    try:
        claims = jwt.decode(token, key, algorithms=["RS256"], audience=client_id, issuer=ISSUER,
                            options={"require": ["exp", "iss", "aud", "sub"]})
    except jwt.PyJWTError:
        raise OAuthError("ID token validation failed") from None
    if nonce is not None and not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
        raise OAuthError("ID token nonce mismatch")
    if expected_subject is not None and claims["sub"] != expected_subject:
        raise OAuthError("Account identity mismatch")
    if claims.get("azp") and claims["azp"] != client_id:
        raise OAuthError("ID token authorized-party mismatch")
    if isinstance(claims.get("aud"), list) and len(claims["aud"]) > 1 and claims.get("azp") != client_id:
        raise OAuthError("Multiple audiences require a matching authorized party")
    return claims


def granted_scopes(tokens):
    value = tokens.get("scope", "")
    if not isinstance(value, str):
        raise OAuthError("Invalid granted scopes")
    return value.split()


def token_profile(tokens, client_id, host_id, claims, previous=None):
    scopes = granted_scopes(tokens)
    if "chatgpt.tokens.use.direct" not in scopes:
        raise OAuthError("ChatGPT plan permission was not granted; no inference enabled")
    if tokens.get("token_type", "").lower() != "bearer" or not tokens.get("access_token"):
        raise OAuthError("Invalid bearer token response")
    profile = dict(previous or {})
    profile.update(email=claims.get("email", ""), subject=claims["sub"], issuer=ISSUER,
                   client_id=client_id, ext_agent_host_id=host_id,
                   access_token=tokens["access_token"],
                   refresh_token=tokens.get("refresh_token", profile.get("refresh_token")),
                   id_token=tokens.get("id_token", profile.get("id_token")), scopes=scopes,
                   expires_at=time.time()+int(tokens["expires_in"]),
                   earliest_refresh_at=tokens.get("earliest_refresh_at"), saved_at=time.time())
    return profile


class CredentialLock:
    def __enter__(self):
        import msvcrt
        PRIVATE.mkdir(exist_ok=True)
        self.f = (PRIVATE/"connection.lock").open("a+b")
        self.f.write(b"0")
        self.f.flush()
        self.f.seek(0)
        try:
            msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.f.close()
            raise OAuthError("Another process is using the OAuth connection") from None
        return self
    def __exit__(self, *_):
        self.f.close()


def access_token():
    # Serialize rotating-token refreshes. Never consume another app's token files.
    with CredentialLock():
        p = read_profile()
        if "chatgpt.tokens.use.direct" not in p["scopes"]:
            raise OAuthError("ChatGPT plan permission is missing")
        if p["expires_at"] > time.time() + 120:
            return p["access_token"]
        if not p.get("refresh_token"):
            raise OAuthError("Session expired; sign in again")
        tokens = http_json(TOKEN, {"grant_type": "refresh_token", "client_id": p["client_id"],
                           "refresh_token": p["refresh_token"], "resource": RESOURCE}, form=True)
        # OAuth refresh may omit scope: retain the existing grant in that case.
        tokens.setdefault("scope", " ".join(p["scopes"]))
        claims = ({"sub": p["subject"], "email": p["email"]} if not tokens.get("id_token") else
                  validate_id_token(tokens["id_token"], p["client_id"], expected_subject=p["subject"]))
        p = token_profile(tokens, p["client_id"], p["ext_agent_host_id"], claims, p)
        save_profile(p)
        return p["access_token"]


def auth_params(host_id, redirect, verifier, state, nonce, client_id=None):
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    params = {"client_id": client_id or "dynamic_agent_client", "ext_agent_host_id": host_id,
              "response_type": "code", "redirect_uri": redirect, "scope": SCOPES,
              "resource": RESOURCE, "state": state, "nonce": nonce,
              "code_challenge_method": "S256", "code_challenge": challenge}
    if not client_id:
        params["agent_name_hint"] = APP_NAME
    return params


def login(port=1455, open_browser=False, ready_file=None):
    # Check dependency BEFORE consent, not after receiving an authorization code.
    import jwt  # noqa: F401
    if os.name != "nt":
        raise OAuthError("Complete this connection in the Windows browser")
    old = read_profile() if (PRIVATE/"connection.dpapi").exists() else None
    redirect = f"http://127.0.0.1:{port}/auth/callback"
    review_path = "/review/" + secrets.token_urlsafe(24)
    csrf = secrets.token_urlsafe(32)
    session = {"started": False, "done": False, "approved": False, "error": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Callback URLs contain credentials: suppress all HTTP access logging.

        def page(self, text, status=200):
            data = ("<!doctype html><html lang='fr'><meta charset='utf-8'><title>" + APP_NAME +
                    "</title><body style='font:18px system-ui;max-width:760px;margin:60px auto;padding:20px'>"+
                    text+"</body></html>").encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            # HTML form POSTs serialize Origin as null under no-referrer (Fetch).
            # Preserve Origin only within this exact loopback review origin;
            # callback pages still suppress referrers, including authorization codes.
            self.send_header("Referrer-Policy", "same-origin" if
                             urllib.parse.urlparse(self.path).path == review_path else "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            return self.headers.get("Host") == f"127.0.0.1:{port}"

        def do_GET(self):
            if not self.allowed():
                return self.page("Hôte refusé", 400)
            uri = urllib.parse.urlparse(self.path)
            if uri.path == review_path:
                account = ("Renouvellement de la connexion existante, même compte et espace." if old else
                           "Nouvelle connexion propre au benchmark; aucun identifiant Hermes ou Codex réutilisé.")
                return self.page(
                    "<h1>Connexion ChatGPT du benchmark</h1><p>À ouvrir dans le navigateur du PC Windows, pas sur iPhone.</p>"
                    "<p>" + account + "</p><p>Cette action demande ton identité de base et l'autorisation d'utiliser ton forfait ChatGPT "
                    "pour des requêtes compatibles. Elle ne donne pas accès à tes conversations ChatGPT.</p>"
                    "<p>Les permissions comprennent une session renouvelable (<code>offline_access</code>), "
                    "<code>resource.invoke</code> et <code>chatgpt.tokens.use.direct</code>.</p>"
                    "<p>Après ton accord sur OpenAI, les jetons seront stockés avec Windows DPAPI CurrentUser dans "
                    "<code>benchmark/private/connection.dpapi</code>. Ils ne seront pas affichés ni envoyés au chat.</p>"
                    "<p>La connexion seule ne lance aucun benchmark. Les appels futurs consommeront les limites de ton forfait "
                    "ou les crédits disponibles selon tes réglages. Tu peux gérer ou révoquer cet accès dans les paramètres ChatGPT.</p>"
                    "<p>En cliquant, tu autorises la création de cette connexion persistante, sous réserve de ton consentement sur OpenAI.</p>"
                    f"<form method='post' action='{review_path}'><input type='hidden' name='csrf' value='{csrf}'>"
                    "<button style='font:inherit;padding:12px'>Continuer avec ChatGPT / Continue with ChatGPT</button></form>"
                )
            if uri.path != "/auth/callback":
                return self.page("Page inconnue", 404)
            if not session["approved"] or not session["started"] or session["done"]:
                return self.page("Aucune tentative active", 400)
            values = urllib.parse.parse_qs(uri.query)
            # Duplicate parameters are rejected to prevent ambiguous callback interpretation.
            if any(len(v) != 1 for v in values.values()):
                return self.page("Retour ambigu refusé", 400)
            if not hmac.compare_digest(values.get("state", [""])[0], session["state"]):
                return self.page("État de connexion refusé", 400)
            if values.get("error"):
                session["done"], session["error"] = True, "Consent declined or authorization failed"
                return self.page("Connexion annulée. Aucun jeton échangé.", 400)
            issued = values.get("client_id", [old["client_id"] if old else ""])[0]
            if not issued or issued == "dynamic_agent_client" or (old and issued != old["client_id"]):
                return self.page("Inscription incomplète ou client différent", 400)
            code = values.get("code", [""])[0]
            if not code:
                return self.page("Code absent", 400)
            session["done"] = True  # Single-use code; no callback auto retry.
            try:
                with CredentialLock():
                    tokens = http_json(TOKEN, {"grant_type": "authorization_code", "client_id": issued,
                        "code": code, "code_verifier": session["verifier"],
                        "redirect_uri": redirect, "resource": RESOURCE}, form=True)
                    claims = validate_id_token(tokens["id_token"], issued, session["nonce"],
                                               old["subject"] if old else None)
                    profile = token_profile(tokens, issued, session["host_id"], claims)
                    save_profile(profile)
                    session["connected"] = True
                self.page("<h1>Connexion confirmée</h1><p>Identité vérifiée; accès au forfait activé pour le benchmark. "
                          "Les jetons sont chiffrés sur Windows. Aucun appel d'inférence effectué.</p>"
                          "<p>Tu peux fermer cet onglet et retourner au chat.</p>")
                print("OAuth connected; identity and granted plan scope validated. No inference performed.", flush=True)
            except Exception:
                # Never log JWT exceptions or token endpoint payloads.
                session["error"] = "Token exchange, identity/scope validation or protected storage failed"
                self.page("Connexion non finalisée : échange, validation ou stockage refusé. Aucun benchmark lancé.", 400)
                print("OAuth not finalized; inspect connection status locally. No secrets logged.", flush=True)

        def do_POST(self):
            if not self.allowed() or self.path != review_path:
                return self.page("Tentative refusée", 400)
            origin = self.headers.get("Origin")
            if origin != f"http://127.0.0.1:{port}":
                return self.page("Origine refusée", 400)
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024:
                return self.page("Requête trop grande", 400)
            body = urllib.parse.parse_qs(self.rfile.read(length).decode())
            if not hmac.compare_digest(body.get("csrf", [""])[0], csrf):
                return self.page("Confirmation refusée", 400)
            if session["started"]:
                # Idempotent recovery: reuse the already approved attempt, never a
                # second client/PKCE/state or an extended authorization deadline.
                target = html.escape(session["authorization_url"], quote=True)
                return self.page("<h1>Connexion déjà démarrée</h1><p>Aucun nouveau flux n'a été créé. "
                    "Poursuis la même tentative auprès d'OpenAI.</p>"
                    f"<p><a href='{target}' rel='noreferrer'>Ouvrir le consentement OpenAI</a></p>")
            session["approved"] = True
            # Host identifier is created only after the user's explicit action-time approval.
            PRIVATE.mkdir(exist_ok=True)
            host_file = PRIVATE/"host.json"
            host_id = json.loads(host_file.read_text())["ext_agent_host_id"] if host_file.exists() else "urn:uuid:"+str(uuid.uuid4())
            if not host_file.exists():
                atomic_bytes(host_file, json.dumps({"ext_agent_host_id": host_id}).encode())
            session.update(started=True, host_id=host_id, state=secrets.token_urlsafe(32),
                           nonce=secrets.token_urlsafe(32), verifier=secrets.token_urlsafe(64))
            params = auth_params(host_id, redirect, session["verifier"], session["state"],
                                 session["nonce"], old["client_id"] if old else None)
            session["authorization_url"] = AUTHORIZE + "?" + urllib.parse.urlencode(params)
            session["authorization_deadline"] = time.time() + AUTHORIZATION_WAIT_SECONDS
            # No ID token hint in the browser URL. User selects their account explicitly.
            self.send_response(303)
            self.send_header("Location", session["authorization_url"])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    url = f"http://127.0.0.1:{port}{review_path}"
    if ready_file:
        # Only an inert local review URL; no OAuth state/code/verifier/token.
        atomic_bytes(Path(ready_file), json.dumps({"review_url": url,
                     "expires_at": time.time()+REVIEW_WAIT_SECONDS,
                     "persistent_access": "pending_user_approval"}).encode())
    print("Open this review page in the Windows browser (not on iPhone): " + url, flush=True)
    if open_browser:
        webbrowser.open(url)
    deadline = time.time() + REVIEW_WAIT_SECONDS
    try:
        while not session["done"]:
            if time.time() >= session.get("authorization_deadline", deadline):
                break
            server.handle_request()
    finally:
        server.server_close()
        connected = session.get("connected", False)
        session.clear()
    print("Local sign-in listener stopped.", flush=True)
    if not connected:
        raise OAuthError("Sign-in cancelled, failed, or timed out; no confirmed connection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["login", "status", "models"])
    parser.add_argument("--port", type=int, default=1455)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    if args.command == "login":
        login(args.port, args.open_browser, args.ready_file)
    elif args.command == "status":
        path = PRIVATE/"connection.dpapi"
        print("Connection present (DPAPI encrypted)" if path.exists() else "Not connected")
    elif args.command == "models":
        catalog = http_json(RESOURCE+"/models", token=access_token())
        models = [{"slug": m["slug"], "display_name": m.get("display_name", m["slug"])}
                  for m in catalog.get("models", []) if m.get("visibility") == "list"]
        print(json.dumps(models, ensure_ascii=False, indent=2))
        print("Catalog only: access to a selected model is verified by successful inference, not this listing.")


if __name__ == "__main__":
    try:
        main()
    except (OAuthError, ImportError, OSError):
        print("OAuth action stopped. Run Setup.ps1 if dependencies are missing; no secret was printed.", file=sys.stderr)
        sys.exit(1)
