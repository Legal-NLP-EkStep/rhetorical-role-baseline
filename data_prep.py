import re


def remove_unwanted_text(text):
    '''Looks for pattern  which typically starts the main text of jugement.
    The text before this pattern contains metadata like name of paries, judges and hence removed'''
    pos_list = []
    len = 0
    pos = 0
    pos_list.append(text.find("JUDGMENT & ORDER"))
    pos_list.append(text.find("J U D G M E N T"))
    pos_list.append(text.find("JUDGMENT"))
    pos_list.append(text.find("O R D E R"))
    pos_list.append(text.find("ORDER"))

    for i, p in enumerate(pos_list):

        if p != -1:
            if i == 0:
                len = 16
            elif i == 1:
                len = 15
            elif i == 2:
                len = 8
            elif i == 3:
                len = 9
            elif i == 4:
                len = 5
            pos = p + len
            break

    return pos


def convert_upper_case_to_title(txt):
    ########### convert the uppercase words to title case for catching names in NER
    title_tokens = []
    for token in txt.split(' '):
        title_subtokens = []
        for subtoken in token.split('\n'):
            if subtoken.isupper():
                title_subtokens.append(subtoken.title())
            else:
                title_subtokens.append(subtoken)
        title_tokens.append('\n'.join(title_subtokens))
    title_txt = ' '.join(title_tokens)
    return title_txt


def guess_preamble_end(truncated_txt, nlp):
    ######### Guess the end of preamble using hueristics
    preamble_end = 0
    truncated_doc = nlp(truncated_txt)
    successive_preamble_pattern_breaks = 0
    preamble_patterns_breaks_theshold = 1  ####### end will be marked after these many consecutive sentences which dont match preamble pattern
    sent_list = [sent for sent in truncated_doc.sents]
    for sent_id, sent in enumerate(sent_list):
        ###### check if verb is present in the sentence
        verb_exclusions = ['reserved', 'pronounced', 'dated', 'signed']
        sent_pos_tag = [token.pos_ for token in sent if token.lower_ not in verb_exclusions]
        verb_present = 'VERB' in sent_pos_tag

        ###### check if uppercase or title case
        allowed_lowercase = ['for', 'at', 'on', 'the', 'in', 'of']
        upppercase_or_titlecase = all(
            [token.text in allowed_lowercase or token.is_upper or token.is_title or token.is_punct for token in sent if
             token.is_alpha])

        if verb_present and not upppercase_or_titlecase:
            successive_preamble_pattern_breaks += 1
            if successive_preamble_pattern_breaks > preamble_patterns_breaks_theshold:
                preamble_end = sent_list[sent_id - preamble_patterns_breaks_theshold - 1].end_char
                break
        else:
            if successive_preamble_pattern_breaks > 0 and (verb_present or not upppercase_or_titlecase):
                preamble_end = sent_list[sent_id - preamble_patterns_breaks_theshold - 1].end_char
                break
            else:
                successive_preamble_pattern_breaks = 0
    return preamble_end


def seperate_and_clean_preamble(txt, preamble_splitting_nlp):
    ########## seperate preamble from judgment text

    ######## get preamble end offset based on keywords
    keyword_preamble_end_offset = remove_unwanted_text(txt)
    if keyword_preamble_end_offset == 0:
        preamble_end_offset = 5000  ######## if keywords not found then set arbitrarty value
    else:
        preamble_end_offset = keyword_preamble_end_offset + 200  ######## take few more characters as judge names are written after JUDGEMENT keywords
    truncated_txt = txt[:preamble_end_offset]
    guessed_preamble_end = guess_preamble_end(truncated_txt, preamble_splitting_nlp)

    if guessed_preamble_end == 0:
        preamble_end = keyword_preamble_end_offset
    else:
        preamble_end = guessed_preamble_end

    preamble_txt = txt[:preamble_end]
    # title_txt = convert_upper_case_to_title(preamble_txt)
    return preamble_txt, preamble_end


def get_spacy_nlp_pipeline_for_indian_legal_text(model_name="en_core_web_sm", disable=['ner'], punc=[".", "?", "!"],
                                                 custom_ner=False):
    ########## Creates spacy nlp pipeline for indian legal text. the sentence splitting is done on specific punctuation marks.
    #########This is finalized after multiple experiments comparison. To use all components pass empty list  disable = []

    import spacy
    from spacy.pipeline import Sentencizer
    try:
        spacy.prefer_gpu()
    except:
        pass
    nlp = spacy.load(model_name, disable=disable)
    nlp.max_length = 30000000

    ############ special tokens which should not be split in tokenization.
    #           this is specially helpful when we use models which split on the dots present in these abbreviations
    # special_tokens_patterns_list = [r'nos?\.',r'v\/?s\.?',r'rs\.',r'sh?ri\.']
    # special_tokens = re.compile( '|'.join(special_tokens_patterns_list),re.IGNORECASE).match
    # nlp.tokenizer = Tokenizer(nlp.vocab,token_match = special_tokens)

    ############## Custom NER patterns
    patterns = [{"label": "RESPONDENT",
                 "pattern": [{"LOWER": "respondent"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(\d+|\,|and|to)"}, "OP": "+"},
                             {"TEXT": {"REGEX": "\d+"}}]},
                {"label": "RESPONDENT",
                 "pattern": [{"LOWER": "respondent"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(^(?i)no\.\d+$)"}, "OP": "*"},
                             {"TEXT": {"REGEX": "(\d+)"}, "OP": "*"}]},
                {"label": "WITNESS",
                 "pattern": [{"TEXT": {"REGEX": r"^(?i)PW\-\d*\w+$"}},
                             {"TEXT": {"REGEX": r"^(\/|[A-Z])$"}, "OP": "*"}]},
                {"label": "WITNESS",
                 "pattern": [{"LOWER": "prosecution"}, {"TEXT": {"REGEX": r"(^(?i)Witness\-\S+$)|(^(?i)witness)"}},
                             {"TEXT": {"REGEX": "(\d+)"}, "OP": "*"},
                             {"TEXT": {"REGEX": r"^(\/|[A-Z])$"}, "OP": "*"}]},
                {"label": "WITNESS",
                 "pattern": [{"TEXT": {"REGEX": "(^(?i)PW$)"}}, {"TEXT": {"REGEX": "(\d+)"}},
                             {"TEXT": {"REGEX": r"^(\/|[A-Z])$"}, "OP": "*"}]},
                {"label": "ACCUSED",
                 "pattern": [{"LOWER": "accused"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(\d+|\,|and|to)"}, "OP": "+"},
                             {"TEXT": {"REGEX": "\d+"}}]},
                {"label": "ACCUSED",
                 "pattern": [{"LOWER": "accused"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(^(?i)no\.\d+$)"}, "OP": "*"},
                             {"TEXT": {"REGEX": "(\d+)"}, "OP": "*"}]}]

    if int(spacy.__version__.split(".")[0]) >= 3:
        ########### For transformer model use built in sentence splitting. For others, use sentence splitting on punctuations.
        ########### This is because transformer sentence spiltting is doing better than the punctuation spiltting
        if model_name != "en_core_web_trf":
            config = {"punct_chars": punc}
            nlp.add_pipe("sentencizer", config=config, before='parser')

        if "ner" not in disable and custom_ner:
            ruler = nlp.add_pipe("entity_ruler", before='ner')
            ruler.add_patterns(patterns)

    else:
        if model_name != "en_core_web_trf":
            sentencizer = Sentencizer(punct_chars=punc)
            nlp.add_pipe(sentencizer, before="parser")
        if "ner" not in disable and custom_ner:
            from spacy.pipeline import EntityRuler
            ruler = EntityRuler(nlp, overwrite_ents=True)
            ruler.add_patterns(patterns)
            nlp.add_pipe(ruler, before='ner')

    return nlp


def attach_short_sentence_boundries_to_next(revised_sentence_boundries, doc_txt):
    ###### this function accepts the list in the format of output of function "extract_relevant_sentences_for_rhetorical_roles" and returns the revised list with shorter sentences attached to next sentence
    min_char_cnt_per_sentence = 5

    concatenated_sentence_boundries = []
    sentences_to_attach_to_next = ()
    for sentence_boundry in revised_sentence_boundries:
        sentence_txt = doc_txt[sentence_boundry[0]: sentence_boundry[1]]
        if not sentence_txt.isspace():  ### sentences containing only spaces , newlines are discarded
            if sentences_to_attach_to_next:
                sentence_start_char = sentences_to_attach_to_next[0]
            else:
                sentence_start_char = sentence_boundry[0]
            sentence_length_char = sentence_boundry[1] - sentence_start_char
            if sentence_length_char > min_char_cnt_per_sentence:
                concatenated_sentence_boundries.append((sentence_start_char, sentence_boundry[1]))
                sentences_to_attach_to_next = ()
            else:
                if not sentences_to_attach_to_next:
                    sentences_to_attach_to_next = sentence_boundry
    return concatenated_sentence_boundries


def remove_unwanted_text(text):
    '''Looks for pattern  which typically starts the main text of jugement.
    The text before this pattern contains metadata like name of paries, judges and hence removed'''
    pos_list = []
    len = 0
    pos = 0
    pos_list.append(text.find("JUDGMENT & ORDER"))
    pos_list.append(text.find("J U D G M E N T"))
    pos_list.append(text.find("JUDGMENT"))
    pos_list.append(text.find("O R D E R"))
    pos_list.append(text.find("ORDER"))

    for i, p in enumerate(pos_list):

        if p != -1:
            if i == 0:
                len = 16
            elif i == 1:
                len = 15
            elif i == 2:
                len = 8
            elif i == 3:
                len = 9
            elif i == 4:
                len = 5
            pos = p + len
            break

    return pos


def split_preamble_judgement(judgment_txt):
    ###### seperates the preamble and judgement text for all courts. It removes the new lines in between  the sentences.  returns 2 texts
    preamble_end = remove_unwanted_text(judgment_txt)
    preamble_removed_txt = judgment_txt[preamble_end:]
    preamble_txt = judgment_txt[:preamble_end]

    ####### remove the new lines which are not after dot or ?. Assumption is that theses would be in between sentence
    preamble_removed_txt = re.sub(r'([^.\"\?])\n+ *', r'\1 ',
                                  preamble_removed_txt)
    return preamble_txt, preamble_removed_txt
