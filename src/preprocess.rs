use regex::{Captures, Regex};
use std::sync::OnceLock;

fn url_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"https?://\S+|www\.\S+").expect("valid URL regex"))
}

fn email_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b").expect("valid email regex"))
}

fn html_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^>]+>").expect("valid html regex"))
}

fn spaces_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s+").expect("valid spaces regex"))
}

fn punctuation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\w\s]").expect("valid punctuation regex"))
}

fn tokenizer_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\w+|[^\w\s]").expect("valid tokenizer regex"))
}

fn model_name_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b([A-Za-z][A-Za-z0-9]*)-(\d[\d.]*)\b").expect("valid model-name regex")
    })
}

fn range_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(\d+)-(\d+)\b").expect("valid range regex"))
}

fn leading_decimal_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(^|[^0-9])(\.\d+)").expect("valid leading decimal regex"))
}

fn negative_leading_decimal_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(^|[^0-9])-\.([0-9]+)").expect("valid negative leading decimal regex")
    })
}

fn currency_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)").expect("valid currency regex")
    })
}

fn percentage_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(-?[\d,]+(?:\.\d+)?)\s*%").expect("valid percent regex"))
}

fn number_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"-?[\d,]+(?:\.\d+)?").expect("valid number regex"))
}

fn sentence_split_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[.!?]+").expect("valid sentence split regex"))
}

#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    pub lowercase: bool,
    pub replace_numbers: bool,
    pub remove_urls: bool,
    pub remove_emails: bool,
    pub remove_html: bool,
    pub remove_punctuation: bool,
    pub remove_extra_whitespace: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            lowercase: true,
            replace_numbers: true,
            remove_urls: true,
            remove_emails: true,
            remove_html: true,
            remove_punctuation: true,
            remove_extra_whitespace: true,
        }
    }
}

impl TextPreprocessor {
    pub fn process(&self, text: &str) -> String {
        let mut out = text.to_string();

        if self.remove_html {
            out = html_regex().replace_all(out.as_str(), " ").into_owned();
        }
        if self.remove_urls {
            out = url_regex().replace_all(out.as_str(), " ").into_owned();
        }
        if self.remove_emails {
            out = email_regex().replace_all(out.as_str(), " ").into_owned();
        }

        out = expand_contractions(out.as_str());
        out = expand_model_names(out.as_str());
        out = normalize_leading_decimals(out.as_str());
        out = expand_currency(out.as_str());
        out = expand_percentages(out.as_str());
        out = expand_ranges(out.as_str());

        if self.replace_numbers {
            out = replace_numbers(out.as_str());
        }
        if self.remove_punctuation {
            out = punctuation_regex()
                .replace_all(out.as_str(), " ")
                .into_owned();
        }
        if self.lowercase {
            out = out.to_lowercase();
        }
        if self.remove_extra_whitespace {
            out = spaces_regex()
                .replace_all(out.as_str(), " ")
                .trim()
                .to_string();
        }

        out
    }
}

pub fn basic_english_tokenize(text: &str) -> Vec<String> {
    tokenizer_regex()
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

pub fn ensure_punctuation(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    match trimmed.chars().last() {
        Some('.' | '!' | '?' | ',' | ';' | ':') => trimmed.to_string(),
        _ => {
            let mut out = trimmed.to_string();
            out.push(',');
            out
        }
    }
}

pub fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    for sentence in sentence_split_regex().split(text) {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(sentence));
            continue;
        }

        let mut current = String::new();
        for word in sentence.split_whitespace() {
            let next_len = if current.is_empty() {
                word.len()
            } else {
                current.len() + word.len() + 1
            };
            if next_len <= max_len {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
            } else {
                if !current.is_empty() {
                    chunks.push(ensure_punctuation(current.as_str()));
                }
                current.clear();
                current.push_str(word);
            }
        }

        if !current.is_empty() {
            chunks.push(ensure_punctuation(current.as_str()));
        }
    }

    if chunks.is_empty() && !text.trim().is_empty() {
        chunks.push(ensure_punctuation(text));
    }

    chunks
}

fn expand_contractions(text: &str) -> String {
    let mut out = text.to_string();
    let replacements = [
        (r"(?i)\bcan't\b", "cannot"),
        (r"(?i)\bwon't\b", "will not"),
        (r"(?i)\blet's\b", "let us"),
        (r"(?i)\b(\w+)n't\b", "$1 not"),
        (r"(?i)\b(\w+)'re\b", "$1 are"),
        (r"(?i)\b(\w+)'ve\b", "$1 have"),
        (r"(?i)\b(\w+)'ll\b", "$1 will"),
        (r"(?i)\b(\w+)'d\b", "$1 would"),
        (r"(?i)\b(\w+)'m\b", "$1 am"),
        (r"(?i)\bit's\b", "it is"),
    ];

    for (pattern, replacement) in replacements {
        let re = Regex::new(pattern).expect("valid contraction regex");
        out = re.replace_all(out.as_str(), replacement).into_owned();
    }

    out
}

fn expand_model_names(text: &str) -> String {
    model_name_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            format!("{} {}", &caps[1], &caps[2])
        })
        .into_owned()
}

fn normalize_leading_decimals(text: &str) -> String {
    let text = negative_leading_decimal_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            format!("{}-0.{}", &caps[1], &caps[2])
        })
        .into_owned();
    leading_decimal_regex()
        .replace_all(text.as_str(), |caps: &Captures<'_>| {
            format!("{}0{}", &caps[1], &caps[2])
        })
        .into_owned()
}

fn expand_ranges(text: &str) -> String {
    range_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            let lo = number_to_words(caps[1].parse().unwrap_or(0));
            let hi = number_to_words(caps[2].parse().unwrap_or(0));
            format!("{lo} to {hi}")
        })
        .into_owned()
}

fn expand_percentages(text: &str) -> String {
    percentage_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            let raw = caps[1].replace(',', "");
            let words = if raw.contains('.') {
                float_to_words(raw.as_str())
            } else {
                number_to_words(raw.parse().unwrap_or(0))
            };
            format!("{words} percent")
        })
        .into_owned()
}

fn expand_currency(text: &str) -> String {
    currency_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            let symbol = &caps[1];
            let amount = caps[2].replace(',', "");
            let unit = match symbol {
                "$" => "dollar",
                "€" => "euro",
                "£" => "pound",
                "¥" => "yen",
                "₹" => "rupee",
                "₩" => "won",
                "₿" => "bitcoin",
                _ => "unit",
            };

            if let Some((whole, frac)) = amount.split_once('.') {
                let whole_value = whole.parse::<i64>().unwrap_or(0);
                let whole_words = number_to_words(whole_value);
                let cents = frac
                    .chars()
                    .take(2)
                    .collect::<String>()
                    .parse::<i64>()
                    .unwrap_or(0);

                let mut result = if whole_value == 1 {
                    format!("{whole_words} {unit}")
                } else {
                    format!("{whole_words} {unit}s")
                };
                if cents > 0 {
                    let cents_words = number_to_words(cents);
                    let suffix = if cents == 1 { "cent" } else { "cents" };
                    result.push_str(format!(" and {cents_words} {suffix}").as_str());
                }
                result
            } else {
                let value = amount.parse::<i64>().unwrap_or(0);
                let words = number_to_words(value);
                if value == 1 {
                    format!("{words} {unit}")
                } else {
                    format!("{words} {unit}s")
                }
            }
        })
        .into_owned()
}

fn replace_numbers(text: &str) -> String {
    number_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            let raw = caps[0].replace(',', "");
            if raw.contains('.') {
                float_to_words(raw.as_str())
            } else {
                number_to_words(raw.parse().unwrap_or(0))
            }
        })
        .into_owned()
}

fn float_to_words(raw: &str) -> String {
    let mut raw = raw.trim();
    let negative = raw.starts_with('-');
    if negative {
        raw = &raw[1..];
    }

    let words = if let Some((int_part, frac_part)) = raw.split_once('.') {
        let int_words = if int_part.is_empty() {
            "zero".to_string()
        } else {
            number_to_words(int_part.parse().unwrap_or(0))
        };
        let digit_words = frac_part
            .chars()
            .filter_map(digit_to_word)
            .collect::<Vec<_>>()
            .join(" ");
        if digit_words.is_empty() {
            int_words
        } else {
            format!("{int_words} point {digit_words}")
        }
    } else {
        number_to_words(raw.parse().unwrap_or(0))
    };

    if negative {
        format!("negative {words}")
    } else {
        words
    }
}

fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 0 {
        return format!("negative {}", number_to_words(-n));
    }

    const SCALES: [&str; 5] = ["", "thousand", "million", "billion", "trillion"];
    let mut parts = Vec::new();
    let mut rem = n as u64;
    let mut scale_idx = 0usize;

    while rem > 0 && scale_idx < SCALES.len() {
        let chunk = (rem % 1000) as u16;
        if chunk != 0 {
            let chunk_words = three_digits_to_words(chunk);
            let scale = SCALES[scale_idx];
            if scale.is_empty() {
                parts.push(chunk_words);
            } else {
                parts.push(format!("{chunk_words} {scale}"));
            }
        }
        rem /= 1000;
        scale_idx += 1;
    }

    if rem > 0 {
        return n.to_string();
    }

    parts.reverse();
    parts.join(" ")
}

fn three_digits_to_words(n: u16) -> String {
    const ONES: [&str; 20] = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    let mut words = Vec::new();
    let hundreds = n / 100;
    let remainder = n % 100;

    if hundreds > 0 {
        words.push(format!("{} hundred", ONES[hundreds as usize]));
    }

    if remainder > 0 {
        if remainder < 20 {
            words.push(ONES[remainder as usize].to_string());
        } else {
            let tens = remainder / 10;
            let ones = remainder % 10;
            if ones > 0 {
                words.push(format!("{}-{}", TENS[tens as usize], ONES[ones as usize]));
            } else {
                words.push(TENS[tens as usize].to_string());
            }
        }
    }

    words.join(" ")
}

fn digit_to_word(ch: char) -> Option<&'static str> {
    match ch {
        '0' => Some("zero"),
        '1' => Some("one"),
        '2' => Some("two"),
        '3' => Some("three"),
        '4' => Some("four"),
        '5' => Some("five"),
        '6' => Some("six"),
        '7' => Some("seven"),
        '8' => Some("eight"),
        '9' => Some("nine"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{TextPreprocessor, basic_english_tokenize, chunk_text};

    #[test]
    fn preprocessor_replaces_numbers() {
        let pp = TextPreprocessor::default();
        let out = pp.process("GPT-3 costs $4.99 and is 50% faster.");
        assert_eq!(
            out,
            "gpt three costs four dollars and ninety nine cents and is fifty percent faster"
        );
    }

    #[test]
    fn tokenizer_keeps_punctuation_tokens() {
        let tokens = basic_english_tokenize("hello, world!");
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn chunking_splits_long_text() {
        let chunks = chunk_text(
            "This sentence is very long and should be split into smaller chunks for inference safety",
            24,
        );
        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|c| c.ends_with(',') || c.ends_with(':') || c.ends_with(';'))
        );
    }
}
