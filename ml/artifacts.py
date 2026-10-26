import hashlib
import hmac
import os
import pickle
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"MATCHLAB1\n"
MAX_BYTES = 25 * 1024 * 1024


def model_dir():
    path = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def signing_key():
    override = os.environ.get("MODEL_SIGNING_KEY")
    if override:
        if len(override) < 32:
            raise ValueError("MODEL_SIGNING_KEY должен содержать не менее 32 символов")
        return override.encode()
    path = model_dir() / "signing.key"
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
    return path.read_bytes()


def dumps(bundle):
    data = pickle.dumps(bundle, protocol=5)
    return MAGIC + hmac.digest(signing_key(), data, "sha256") + data


def loads(data):
    if len(data) > MAX_BYTES or not data.startswith(MAGIC):
        raise ValueError("Нужен подписанный .pkl, созданный этим проектом")
    signature, payload = data[len(MAGIC) : len(MAGIC) + 32], data[len(MAGIC) + 32 :]
    if not hmac.compare_digest(signature, hmac.digest(signing_key(), payload, "sha256")):
        raise ValueError("Подпись модели не совпадает; небезопасный pickle отклонён")
    obj = pickle.loads(payload)
    if not isinstance(obj, dict) or obj.get("format") != 1 or "pipeline" not in obj:
        raise ValueError("Неподдерживаемый формат модели")
    return obj


def save(bundle, name="champion.pkl"):
    target = model_dir() / name
    temp = target.with_suffix(".tmp")
    temp.write_bytes(dumps(bundle))
    temp.replace(target)
    return target


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()
