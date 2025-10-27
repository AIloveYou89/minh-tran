# preprocess.py - Optimized Version
import re
from num2words import num2words
from typing import List

# Tối ưu chunk size cho tốc độ xử lý
MIN_CHARS_PER_CHUNK = 50  # Giảm từ 150 để xử lý nhanh hơn
MAX_CHARS_PER_CHUNK = 130  # Giảm từ 250 để tránh token limit
OPTIMAL_CHUNK_SIZE = 80   # Sweet spot cho model
PUNCS = r".?!…"

# Cache compiled regex patterns để tăng tốc
_number_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*)(?:\s*(%|[^\W\d_]+))?", re.UNICODE)
_whitespace_pattern = re.compile(r"\s+")
_comma_pattern = re.compile(r"\s*,\s*")
_punct_spacing_pattern = re.compile(r"\s+([,;:])")
_repeated_punct_pattern = re.compile(rf"[{PUNCS}]{{2,}}")
_punct_no_space_pattern = re.compile(rf"([{PUNCS}])(?=\S)")

def normalize_text_vn(text: str) -> str:
    """Tối ưu normalize với cached regex patterns"""
    text = text.strip()
    text = _whitespace_pattern.sub(" ", text)
    text = _comma_pattern.sub(", ", text)
    text = text.lower()
    
    def repl_number_with_unit(m):
        num_str = m.group(1).replace(".", "")
        unit = m.group(2) or ""
        try:
            return num2words(int(num_str), lang="vi") + (" " + unit if unit else "")
        except:
            return m.group(0)
    
    text = _number_pattern.sub(repl_number_with_unit, text)
    text = _punct_spacing_pattern.sub(r"\1", text)
    text = _repeated_punct_pattern.sub(lambda m: m.group(0)[0], text)
    text = _punct_no_space_pattern.sub(r"\1 ", text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Chia câu với regex tối ưu"""
    parts = re.split(rf"(?<=[{PUNCS}])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(rf"[{PUNCS}]+", p):
            continue
        out.append(p)
    return out

def ensure_punctuation(s: str) -> str:
    """Đảm bảo câu có dấu câu"""
    s = s.strip()
    if not s.endswith(tuple(PUNCS)):
        s += "."
    return s

def ensure_leading_dot(s: str) -> str:
    """Đảm bảo câu bắt đầu bằng dấu chấm nếu cần"""
    s = s.lstrip()
    if s and s[0] not in PUNCS:
        return ". " + s
    return s

def smart_chunk_split(text: str) -> List[str]:
    """Chia chunk thông minh dựa trên độ dài optimal"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        
        if current_length + word_len > MAX_CHARS_PER_CHUNK and current_chunk:
            # Kiểm tra nếu chunk quá ngắn, thêm thêm từ
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) < MIN_CHARS_PER_CHUNK and len(chunks) > 0:
                # Merge với chunk trước nếu có thể
                prev_chunk = chunks[-1]
                if len(prev_chunk) + len(chunk_text) + 1 <= MAX_CHARS_PER_CHUNK:
                    chunks[-1] = prev_chunk + " " + chunk_text
                    current_chunk = [word]
                    current_length = word_len
                    continue
            
            chunks.append(chunk_text)
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) < MIN_CHARS_PER_CHUNK and chunks:
            chunks[-1] += " " + chunk_text
        else:
            chunks.append(chunk_text)
    
    return chunks

def split_long_text(text: str, max_len=MAX_CHARS_PER_CHUNK) -> List[str]:
    """Chia văn bản dài thành các phần nhỏ hơn"""
    parts = []
    t = text.strip()
    while len(t) > max_len:
        split_pos = t.rfind(" ", max_len - 50, max_len)
        if split_pos == -1:
            split_pos = max_len
        parts.append(t[:split_pos].strip())
        t = t[split_pos:].strip()
    if t:
        parts.append(t)
    return parts

def preprocess_text(text: str) -> List[str]:
    """Preprocessing tối ưu với chunk size thông minh - MAIN FUNCTION"""
    clean = normalize_text_vn(text)
    sentences = split_into_sentences(clean)
    
    if not sentences:
        s = ensure_punctuation(clean)
        return [ensure_leading_dot(s)]
    
    # Ưu tiên ghép câu để đạt optimal chunk size
    chunks = []
    buffer = ""
    
    for i, sent in enumerate(sentences, 1):
        sent = ensure_punctuation(sent)
        sent = re.sub(r'^([^\w]*\w[^,]{0,10}),\s*', r'\1 ', sent)
        
        # Xử lý 5 câu đầu riêng biệt như code gốc
        if i <= 5:
            if len(sent) > MAX_CHARS_PER_CHUNK:
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                chunks.append(ensure_leading_dot(sent))
        else:
            # Nếu câu quá dài, chia nhỏ
            if len(sent) > MAX_CHARS_PER_CHUNK:
                if buffer:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = ""
                chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(sent)])
            else:
                # Thêm vào buffer nếu không vượt quá optimal size
                if buffer and len(buffer) + len(sent) + 1 > OPTIMAL_CHUNK_SIZE:
                    chunks.append(ensure_leading_dot(buffer))
                    buffer = sent
                elif buffer:
                    buffer += " " + sent
                else:
                    buffer = sent
    
    if buffer:
        if len(buffer) > MAX_CHARS_PER_CHUNK:
            chunks.extend([ensure_leading_dot(s) for s in smart_chunk_split(buffer)])
        else:
            chunks.append(ensure_leading_dot(ensure_punctuation(buffer)))
    
    return chunks

def get_chunk_stats(chunks: List[str]) -> dict:
    """Thống kê chunks để debug"""
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "optimal_ratio": sum(1 for l in lengths if MIN_CHARS_PER_CHUNK <= l <= OPTIMAL_CHUNK_SIZE) / len(lengths) if lengths else 0
    }

# Test nhanh
if __name__ == "__main__":
    sample = "Audio 360, nếu thấy hay thì like nhé! 10.000 tệ là phần thưởng. Đây là một đoạn văn bản dài hơn để test chunking. Chúng ta sẽ xem liệu nó có được chia đúng cách không."
    
    print("=== TEST PREPROCESS ===")
    chunks = preprocess_text(sample)
    stats = get_chunk_stats(chunks)
    
    print(f"Input length: {len(sample)} chars")
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Length range: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.1f})")
    print(f"Optimal ratio: {stats['optimal_ratio']:.1%}")
    print()
    
    for idx, c in enumerate(chunks, 1):
        print(f"{idx}. ({len(c)} chars) {repr(c)}")
