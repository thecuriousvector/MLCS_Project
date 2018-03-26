'''
A dictionary that stores the one-on-one correspondence between 321 gender specific
words/phrases and whether they indicate male or female. 1 stands for male and -1 stands for
female. The classification of the gender specific worsd are from the following three lists:

1. Gender in English: Masculine and Feminine Words, ilu English:
http://www.iluenglish.com/gender-in-english-masculine-and-feminine-words/

2. Stjerneskinn.com: http://stjerneskinn.com/gender-neutral-words.htm

3. EXAMPLES OF GENDER-SENSITIVE LANGUAGE: 
http://www.servicegrowth.net/documents/Examples%20of%20Gender-Sensitive%20Language.net.pdf
'''
gender_dictionary = {}

'''
Read words from two lists, separately a list of male words and a list of female words -
female_list and a list of male words - male_list. For a word in male_list, the value
associated with it in the dictionary is 1 and for a word in female_list, the value
associated in the dictionary is -1.
'''
male_list = open('male_list.txt', 'r')
female_list = open('female_list.txt', 'r')

for word in male_list:
	male_key = word.rstrip('\n')
	gender_dictionary[male_key] = 1

for word in female_list:
	female_key = word.rstrip('\n')
	gender_dictionary[female_key] = -1

male_list.close()
female_list.close()

def G(word):
    '''
    A function that accepts a string as an argument and returns the gender associated with 
    the string.

    Args
    ----
    word - a str

    Returns
    -------
    1  - if the word is in gender_dictionary and gender_dictionary[word] = 1
    -1 - if the word is in gender_dictionary and gender_dictioanry[word] = -1
    0  - if the word is not in gender_dictionary
    '''
    return gender_dictionary.get(word, 0)
