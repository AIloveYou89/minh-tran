# preprocess.py
import re
from num2words import num2words

MIN_CHARS_PER_CHUNK = 150  # các chunk sau câu thứ 5 sẽ >= 150 ký tự
MAX_CHARS_PER_CHUNK = 250  # các chunk sau câu thứ 5 sẽ không quá 250 ký tự

def normalize_text_vn(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    def repl_number(m):
        try:
            return num2words(int(m.group()), lang="vi")
        except:
            return m.group()

    text = re.sub(r"\d+", repl_number, text)
    text = re.sub(r'([.?!])(?=\S)', r'\1 ', text)
    return text.strip()

def split_into_sentences(text: str) -> list[str]:
    """
    Tách văn bản thành các câu theo . ? ! (giữ lại dấu).
    """
    parts = re.split(r'(?<=[.?!])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def ensure_punctuation(s: str) -> str:
    s = s.strip()
    if not s.endswith((".", "?", "!")):
        s += "."
    return s

def ensure_leading_dot(s: str) -> str:
    s = s.lstrip()
    if s and s[0] not in ".!?":
        return "." + s
    return s

def split_long_text(text: str, max_len=MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    Cắt đoạn text dài thành các phần nhỏ <= max_len.
    """
    parts = []
    while len(text) > max_len:
        split_pos = text.rfind(" ", max_len - 50, max_len)  # cố gắng cắt ở khoảng trắng
        if split_pos == -1:
            split_pos = max_len
        parts.append(text[:split_pos].strip())
        text = text[split_pos:].strip()
    if text:
        parts.append(text)
    return parts

def preprocess_text(text: str) -> list[str]:
    clean = normalize_text_vn(text)
    sentences = split_into_sentences(clean)

    if not sentences:
        s = ensure_punctuation(clean)
        return [ensure_leading_dot(s)]

    chunks = []
    buffer = ""

    for i, sent in enumerate(sentences, 1):
        sent = ensure_punctuation(sent)

        if i <= 5:
            # 5 câu đầu → mỗi câu 1 chunk
            if len(sent) > MAX_CHARS_PER_CHUNK:
                chunks.extend([ensure_leading_dot(s) for s in split_long_text(sent)])
            else:
                chunks.append(ensure_leading_dot(sent))
        else:
            # Gom các câu thành chunk 150-250 ký tự
            if len(buffer) + len(sent) + 1 <= MAX_CHARS_PER_CHUNK:
                buffer += " " + sent if buffer else sent
            else:
                # Flush buffer nếu đạt đủ min
                if buffer:
                    if len(buffer) > MAX_CHARS_PER_CHUNK:
                        chunks.extend([ensure_leading_dot(s) for s in split_long_text(buffer)])
                    else:
                        chunks.append(ensure_leading_dot(ensure_punctuation(buffer)))
                buffer = sent

    if buffer:
        if len(buffer) > MAX_CHARS_PER_CHUNK:
            chunks.extend([ensure_leading_dot(s) for s in split_long_text(buffer)])
        else:
            chunks.append(ensure_leading_dot(ensure_punctuation(buffer)))

    return chunks

# ——————————— Test nhanh ———————————
if __name__ == "__main__":
    sample = (
        "Diệp Nguyệt Tuyết khinh bỉ liếc nhìn người đàn ông trước mặt, đôi mắt trong "
        "veo căng tràn cơn giận đỏ rực, ngọn lửa ấy đủ sức thiêu đốt cả Tương Vũ. "
        "Câu nói từng chữ đều đầy quyết liệt: Trả lại y phục cho ta! "
        "Mặt nàng ửng hồng, xen lẫn sự bối rối và tức giận. Tương Vũ không những từ chối "
        "mà còn cố tình ôm ấp, vuốt ve chiếc áo nàng đang mặc. Diệp Nguyệt Tuyết cảm giác "
        "như đầu mình sắp nổ tung mất! Càng nhìn, Tương Vũ càng thấy nàng không đơn thuần "
        "là một người phụ nữ bình thường. Đôi mắt ấy sâu thẳm tâm sự, môi đỏ mọng phảng phất "
        "vẻ ngây thơ. Làn da trắng mềm mại, mịn màng như tuyết mới rơi, khiến hắn tin chắc "
        "nàng là một tiên nữ trần gian. Nàng không tin Tương Vũ sẽ trả y phục cho mình, nhất "
        "là khi hiện giờ nàng gần như không mảnh vải che thân, lớp áo mỏng trong suốt lộ rõ "
        "mọi thứ. Cảm giác bất an dâng lên từng khoảnh khắc. Diệp Nguyệt Tuyết quyết định "
        "mạo hiểm, tay kia vội bắt lấy tấm rèm che, giật mạnh rồi nhanh chóng đứng lên, quấn "
        "rèm che quanh người. Tương Vũ ngơ ngác, không nghĩ nàng sẽ táo bạo như vậy. Hương thơm "
        "thoang thoảng của thiếu nữ bỗng quấn lấy mũi hắn, rồi một luồng gió nhẹ thổi qua, cố "
        "đoạt lại chiếc áo trong tay. Diệp Nguyệt Tuyết không ngờ sẽ bị vải thô vướng chân và "
        "ngã về phía trước. Chẳng may tấm rèm dài hơn dự tính."
    )
    for idx, c in enumerate(preprocess_text(sample), 1):
        print(f"{idx}. ({len(c)} chars) {repr(c)}")
