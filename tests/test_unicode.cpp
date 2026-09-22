#include <string>
#include <vector>

#include "check.h"
#include "unicode.h"

using namespace onebit::unicode;

static std::vector<char32_t> v(std::u32string s) { return {s.begin(), s.end()}; }

static void test_utf8() {
    auto ok = decode_utf8("a\xC3\xA9\xE4\xBD\xA0\xF0\x9F\x98\x80");
    REQUIRE(ok.has_value());
    CHECK((*ok == v(U"aé你\U0001F600")));
    CHECK(encode_utf8(*ok) == "a\xC3\xA9\xE4\xBD\xA0\xF0\x9F\x98\x80");
    CHECK(!decode_utf8("\xC0\x80").has_value());          // overlong NUL
    CHECK(!decode_utf8("\xED\xA0\x80").has_value());      // surrogate
    CHECK(!decode_utf8("\xF4\x90\x80\x80").has_value());  // > U+10FFFF
    CHECK(!decode_utf8("\xE4\xBD").has_value());          // truncated
    CHECK(!decode_utf8("\x80").has_value());              // stray continuation
    CHECK(decode_utf8("")->empty());
}

static void test_classes() {
    CHECK(is_letter(U'a') && is_letter(U'é') && is_letter(U'你') && is_letter(U'가'));
    CHECK(!is_letter(U'1') && !is_letter(U' ') && !is_letter(U'!') && !is_letter(U'́'));
    CHECK(is_number(U'7') && is_number(U'٣') && is_number(U'Ⅷ') && is_number(U'½'));
    CHECK(!is_number(U'x'));
    CHECK(is_whitespace(U' ') && is_whitespace(U'\n') && is_whitespace(U' ') && is_whitespace(U'　'));
    CHECK(!is_whitespace(U'​'));  // zero-width space is not White_Space
}

static void test_nfc() {
    CHECK((nfc(v(U"é")) == v(U"é")));                    // e + acute -> é
    CHECK((nfc(v(U"é")) == v(U"é")));                     // already composed
    CHECK((nfc(v(U"각")) == v(U"각")));         // Hangul L V T -> 각
    CHECK((nfc(v(U"ậ")) == v(U"ậ")));              // ậ -> ậ
    CHECK((nfc(v(U"ậ")) == v(U"ậ")));              // reordered marks, same result
    CHECK((nfc(v(U"Å")) == v(U"Å")));                     // Angstrom sign -> Å (singleton)
    CHECK((nfc(v(U"̈́")) == v(U"̈́")));               // excluded from composition
    CHECK((nfc(v(U"q̣̇")) == v(U"q̣̇")));       // no composite: canonical order only
    CHECK((nfc(v(U"hello")) == v(U"hello")));
}

int main() {
    test_utf8();
    test_classes();
    test_nfc();
    return test_result("test_unicode");
}
