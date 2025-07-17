# preprocess.py
import re
from num2words import num2words


def normalize_text_vn(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
      - Loại bỏ khoảng trắng thừa, viết thường
      - Chuyển số thành chữ
      - Đảm bảo khoảng trắng sau dấu ., ?, !
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text).lower()
    def repl_number(m):
        return num2words(int(m.group()), lang="vi")
    text = re.sub(r"\d+", repl_number, text)
    text = re.sub(r'([.?!])(?=\S)', r'\1 ', text)
    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Tách văn bản thành câu, giữ lại dấu kết câu.
    """
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def preprocess_text(text: str) -> list[str]:
    """
    1) Normalize
    2) Split thành câu
    3) Prefix mỗi câu với "." để đảm bảo ngữ điệu
    """
    clean = normalize_text_vn(text)
    sentences = split_into_sentences(clean)
    if not sentences:
        return ["." + clean]
    return ["." + s for s in sentences]