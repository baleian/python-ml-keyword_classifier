BLANK = u'\u0000'
NIL = u'\u0001'

CHOSUNG_LIST = [
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', 
    u'ㅃ', u'ㅅ', u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', 
    u'ㅌ', u'ㅍ', u'ㅎ'
]

JUNGSUNG_LIST = [
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', 
    u'ㅗ', u'ㅘ', u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', 
    u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ'
]

JONGSUNG_LIST = [
    NIL, u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ',
    u'ㄹ', u'ㄺ', u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', 
    u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ', u'ㅆ', u'o', u'ㅈ', u'ㅊ', 
    u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
]

VALID_CHARS = (
    [BLANK, NIL] +  # 예외 문자
    [chr(c) for c in range(ord('ㄱ'), ord('ㅣ') + 1)] +  # 한글 자모
    [chr(c) for c in range(ord('!'), ord('~') + 1)]  # 특문 + 영문자
)

VALID_CHARS_SET = set(VALID_CHARS)

CHAR_INDICES = {c: i for i, c in enumerate(VALID_CHARS)}


def char_2_jaso(c):
    # 한글의 경우 초성,중성,종성으로 split
    if u'가' <= c <= u'힣':
        code = ord(c) - 44032
        char1 = int(code / 588)
        char2 = int((code - (588 * char1)) / 28)
        char3 = int((code - (588 * char1) - (28 * char2)))
        return [CHOSUNG_LIST[char1], JUNGSUNG_LIST[char2], JONGSUNG_LIST[char3]]
    # 영문,숫자,특수문자의 경우 한글과 동일하게 길이 3으로 expand
    elif c in VALID_CHARS_SET:
        return [c, NIL, NIL]
    # invalid 입력에 대해 예외문자 처리
    else:
        return [NIL, NIL, NIL]


def text_2_jaso(text):
    return [jaso for c in text for jaso in char_2_jaso(c)]


def transform(text, max_len=15):
    text = text[0:max_len]
    jaso_list = text_2_jaso(text)
    vector = [CHAR_INDICES[c] for c in jaso_list]
    vector = vector[0:max_len*3]
    vector = vector + ([0]*(max_len*3-len(vector)))
    return vector
