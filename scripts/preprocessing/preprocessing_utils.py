import string
from typing import Dict

import emoji
from preprocessing_parameters import USERNAME_REGEX, URL_REGEX


def replace_emojis(tweet: str, mapping: Dict[str, str], ) -> str:
    """
    Replaces emojis in tweet with their textual representation.
    Textual representation are taken from mapping Dict
    :param tweet: A tweet's text
    :param mapping: Dictionary {emoji : textual representation}
    :return: Modified string with no emojis
    """
    all_emojis_list = emoji.UNICODE_EMOJI
    for emoji_key, word in mapping.items():
        new_tweet = tweet.replace(emoji_key, word)
    for emoji_key in all_emojis_list:
        new_tweet = tweet.replace(emoji_key, ' ')

    return new_tweet


def mask_username(tweet: str) -> str:
    """
    Replaces user mentions with the mask @username
    :param tweet: Str that may contain user mentions
    :return: Modified string
    """
    masked_tweet = USERNAME_REGEX.sub('@username', tweet)
    return masked_tweet


def replace_amps_and_links(tweet: str, repl: str) -> str:
    """
    Replaces "&amp;" with the parameter repl and masks urls
    :param tweet: Tweet's text
    :param repl: String that replaces ampersand's html code
    :return: Modified string
    """
    preprocessed_tweet = tweet.replace('&amp;', repl)
    preprocessed_tweet = URL_REGEX.sub('link', preprocessed_tweet)
    return preprocessed_tweet


def preprocess_tweet_text(tweet: str, emoji_mapping: Dict[str, str], amp_repl: str):
    """
    Facade function for preprocessing functions. This function takes tweet's text
    and masks its user mentions, urls; replaces html codes of '&' with amp_repl
    str; replaces emojies with other string based on emoji_mapping Dict
    :param tweet: Tweet's text
    :param emoji_mapping: Dictionary {emoji : textual representation}
    :param amp_repl: String that replaces ampersand's html code
    :return: Preprocessed tweet's text
    """
    tweet = str(tweet)
    new_tweet = replace_emojis(tweet=tweet, mapping=emoji_mapping)
    new_tweet = replace_amps_and_links(tweet=new_tweet, repl=amp_repl)
    new_tweet = mask_username(tweet=new_tweet, )
    return new_tweet


def clean_text(text: str) -> str:
    text = list_replace('\u00C4', 'A', text)
    text = list_replace('é', 'e', text)
    text = list_replace('è', 'e', text)
    text = list_replace('\u00E4', 'a', text)
    text = text.replace('l’', ' ')
    text = list_replace('\u00CB', 'E', text)
    text = list_replace('\u00EB', 'e', text)
    text = list_replace('\u1E26', 'H', text)
    text = list_replace('\u1E27', 'h', text)
    text = list_replace('\u00CF', 'I', text)
    text = list_replace('\u00EF', 'i', text)
    text = list_replace('\u00D6', 'O', text)
    text = list_replace('\u00F6', 'o', text)
    text = list_replace('\u00DC', 'U', text)
    text = list_replace('\u00FC', 'u', text)
    text = list_replace('\u0178', 'Y', text)
    text = list_replace('\u00FF', 'y', text)
    text = list_replace('\u00DF', 's', text)
    text = list_replace('\u1E9E', 'S', text)
    alphabet = list \
            (
            '\n абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ0123456789§"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

    alphabet.append("'")
    alphabet = set(alphabet)
    alphabet = alphabet.difference(set(string.punctuation))
    cleaned_text = [sym if sym in alphabet else ' ' for sym in text]
    cleaned_text = ''.join(cleaned_text)
    return cleaned_text


def list_replace(search, replacement, text):
    search = [el for el in search if el in text]
    for c in search:
        text = text.replace(c, replacement)
    return text
