/// Counts the approximate number of tokens in a string.
///
/// This function provides a rough estimate based on the assumption that
/// one token is approximately equal to four characters in English text.
/// It may not be accurate for text in other languages or for text that
/// includes a lot of punctuation or special characters.
///
/// # Arguments
///
/// * `text` - A string slice that holds the text to be tokenized.
///
/// # Returns
///
/// * An usize representing the approximate number of tokens in `text`.
pub fn count_tokens(text: &str) -> usize {
    let char_count = text.chars().count();
    char_count / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens() {
        assert_eq!(count_tokens("Hello, world!"), 3);
        assert_eq!(
            count_tokens("This is a longer sentence with more tokens."),
            10
        );
        assert_eq!(count_tokens(""), 0);
    }
}
