import json
import time
from typing import Dict, List
import pdb, sys
import openai
from tqdm import tqdm


def levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def intertwine(m1: List[str], m2: List[str], names: List[str]) -> str:
    # Intertwines two list of strings.
    assert len(m1) == len(m2)
    res = ""
    for i, ut in enumerate(m1):
        res = res + names[0] + ": " + ut + "\n"
        res = res + names[0] + ": " + m2[i] + "\n"
    return res.rstrip()


def parse_prompt(prompt: str, names: List[str]) -> str:
    """
    Read the prompt and split the utterances
    according to names.
    Returns two lists.
    """
    ls = prompt.split("\n")
    r1, r2 = [], []
    for line in ls:
        if line.startswith(names[0]):
            r1.append(line)
        elif line.startswith(names[1]):
            r2.append(line)
        else:
            raise ValueError(f"The sequence '{line}' does not start with {names}.")
    return (r1, r2)


# Prepend fewshot prompt to every element
def prepend_prefix(ls: List[str], prefix: str, sep="") -> List[str]:
    return ["\n\n".join([prefix, sep, el]) for el in ls]


# Strip off the prefix
def remove_prefix(ls: List[str], prefix: str, sep="") -> List[str]:
    return [el.split(sep)[-1].lstrip() for el in ls]


# Append suffices every element
def append_suffix(ls: List[str], suffix: str, sep="") -> List[str]:
    return [sep.join([el, suffix]) for el in ls]


def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like
    spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return out_string


def save_to_json(d: Dict[str, str], path: str):
    with open(path, "w") as f:
        for item in d:
            f.write(json.dumps(item) + "\n")


def chunks(ls, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(ls), n):
        yield ls[i : min(i + n, len(ls))]


def get_generations_gpt3(
    ls: List[str],
    model_name: str,
    clean_tok: bool,
    stop: List[str],
    temperature: float,
    batch_size: int,
    max_length: int,
    penalty: float,
    n: int,
    keyfile: str,
    top_p: float = 1.0,
) -> List[str]:

    openai.api_key = [el for el in open(keyfile, "r")][0][:-1]
    gens = []
    chunks_ls = list(chunks(ls, batch_size))
    for chunk in tqdm(chunks_ls, total=len(chunks_ls)):
        # create a completion
        lst = [el.rstrip(" ") for el in chunk]
        success = False
        retries = 1
        while not success and retries < 200:
            try:
                completion = openai.Completion.create(
                    engine=model_name,
                    prompt=lst,
                    max_tokens=max_length,
                    temperature=temperature,
                    n=n,
                    top_p=top_p,
                    stop=stop,
                    frequency_penalty=penalty,
                )
                success = True
            except Exception as e:
                wait = retries * 10
                print(f'Error, rate limit reached! Waiting {str(wait)} secs and re-trying...')
                sys.stdout.flush()
                time.sleep(wait)
                retries += 1

        # Process the completions
        comps = [c.text for c in completion.choices]
        if clean_tok:
            comps = [clean_up_tokenization(c) for c in comps]
        gens.extend(comps)

    gens = [gen.replace("\xa0", " ").strip() for gen in gens]
    return gens


# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings_gpt3(
    texts: List[str], keyfile: str, engine="text-similarity-davinci-001", batch_size=18
) -> List[List[float]]:
    openai.api_key = [el for el in open(keyfile, "r")][0]
    chunks_ls = list(chunks(texts, batch_size))
    rr = []
    for chunk in tqdm(chunks_ls, total=len(chunks_ls)):
        # Replace newlines, which can negatively affect performance.
        chunk = [str(text).replace("\n", " ") for text in chunk]
        try:
            results = openai.Embedding.create(input=chunk, engine=engine)["data"]
            results = [result["embedding"] for result in results]
            rr.extend(results)
        except Exception as e:
            print(e)
            time.sleep(60)
            results = openai.Embedding.create(input=chunk, engine=engine)["data"]
            results = [result["embedding"] for result in results]
            rr.extend(results)
    return rr

