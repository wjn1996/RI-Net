import difflib
import Levenshtein


def get_equal_rate(str1, str2):
    score = Levenshtein.ratio(str1, str2)
    if score > 0.80:
        return True
    else:
        return False



# def get_equal_rate(str1, str2):
#     score = difflib.SequenceMatcher(None, str1, str2).quick_ratio()
#     if score > 0.85:
#         return True
#     else:
#         return False


if __name__ == '__main__':
    get_equal_rate('Beyoncé', 'Beyonce')