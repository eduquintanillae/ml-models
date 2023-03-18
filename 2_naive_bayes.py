from typing import Set
import re
from typing import NamedTuple
from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict
import glob, re
import random
from utils.machine_learning import split_data
from collections import Counter

def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

def drop_final_s(word):
    return re.sub("s$", "", word)

class Message(NamedTuple):
    text: str
    is_spam: bool

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | not spam)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # We need to consider prior probabilities
        total_messages = self.spam_messages + self.ham_messages
        if total_messages > 0:
            spam_prior = self.spam_messages / total_messages
            ham_prior = self.ham_messages / total_messages
            
            if spam_prior > 0:
                log_prob_if_spam += math.log(spam_prior)
            if ham_prior > 0:
                log_prob_if_ham += math.log(ham_prior)

        # Only iterate over tokens that appear in the text
        for token in text_tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_ham += math.log(prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


def main():        
    random.seed(42)
    path = 'data/spam_ham/*'
    
    data: List[Message] = []
    
    for filename in glob.glob(path):
        is_spam = "ham" not in filename
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break
    
    
    train_messages, test_messages = split_data(data, 0.75)
    
    model = NaiveBayesClassifier()
    model.train(train_messages)
    
    predictions = [(message, model.predict(message.text))
                   for message in test_messages]
    
    print("predictions", predictions)
    
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                               for message, spam_probability in predictions)
    print(confusion_matrix)
    
    def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
        prob_if_spam, prob_if_ham = model._probabilities(token)
    
        return prob_if_spam / (prob_if_spam + prob_if_ham)
    
    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))
    
    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])
    
if __name__ == "__main__": 
    main()