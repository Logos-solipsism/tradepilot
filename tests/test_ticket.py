"""票据：与 Java InternalTicketSigner 逐字同格式的签发与校验。"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from app.security import ticket as ticket_mod
from app.security.ticket import TicketError, sign, verify

SECRET = "test-secret"


@pytest.fixture(autouse=True)
def _clean_nonces():
    ticket_mod.reset_nonce_cache()
    yield
    ticket_mod.reset_nonce_cache()


def test_sign_then_verify_returns_user_id():
    t = sign(42, secret=SECRET, ts=int(time.time()), nonce="n1")
    assert verify(t, secret=SECRET) == 42


def test_ticket_format_is_four_parts_with_64_lower_hex():
    t = sign(7, secret=SECRET, ts=1_700_000_000, nonce="abc123")
    parts = t.split(".")
    assert len(parts) == 4
    assert parts[0] == "7"
    assert parts[1] == "1700000000"
    assert parts[2] == "abc123"
    assert len(parts[3]) == 64
    assert parts[3] == parts[3].lower()
    int(parts[3], 16)  # 合法十六进制


def test_signature_matches_java_payload_without_trailing_newline():
    # 与 Java payload(userId, ts, nonce) 完全一致：三段 \n 连接，末尾无换行
    payload = "42\n1700000000\nn1"
    expected = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    t = sign(42, secret=SECRET, ts=1_700_000_000, nonce="n1")
    assert t.split(".")[3] == expected


def test_plus_300_seconds_passes():
    now = 1_700_000_000
    t = sign(1, secret=SECRET, ts=now - 300, nonce="edge-ok")
    assert verify(t, secret=SECRET, now=now) == 1


def test_plus_301_seconds_rejected():
    now = 1_700_000_000
    t = sign(1, secret=SECRET, ts=now + 301, nonce="edge-bad")
    with pytest.raises(TicketError):
        verify(t, secret=SECRET, now=now)


def test_tampered_signature_rejected():
    t = sign(1, secret=SECRET, ts=int(time.time()), nonce="tamper")
    parts = t.split(".")
    last = parts[3]
    parts[3] = ("0" if last[0] != "0" else "1") + last[1:]
    with pytest.raises(TicketError):
        verify(".".join(parts), secret=SECRET)


def test_empty_secret_fails_closed_on_sign():
    with pytest.raises(TicketError):
        sign(1, secret="", ts=int(time.time()), nonce="x")


def test_empty_secret_fails_closed_on_verify():
    t = sign(1, secret=SECRET, ts=int(time.time()), nonce="y")
    with pytest.raises(TicketError):
        verify(t, secret="   ")


def test_missing_or_malformed_ticket_rejected():
    with pytest.raises(TicketError):
        verify(None, secret=SECRET)
    with pytest.raises(TicketError):
        verify("only.three.parts", secret=SECRET)


def test_nonce_replay_rejected():
    ts = int(time.time())
    t = sign(9, secret=SECRET, ts=ts, nonce="dup")
    assert verify(t, secret=SECRET, now=ts) == 9
    with pytest.raises(TicketError):
        # 同一 nonce 第二次出现一律拒绝
        verify(t, secret=SECRET, now=ts)
