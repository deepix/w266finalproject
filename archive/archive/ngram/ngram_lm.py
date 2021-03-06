from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict
import numpy as np

class BaseLM(object):
    """Base class for n-gram language models.

    Implements some shared functionality.

    You do not (and should not) need to modify this class, but we do encourage
    reading through the implementations of sample_next() and score_seq().
    """
    # Markov order (context size + 1)
    order_n = 0
    # Context counts, as nested map. Outer key is (n-1) word context, inner
    # key is word, inner value is integer count.
    counts = {}
    # Words known to the model, as a list of strings.
    words = []
    # A list of state variables of this model. Used to test equality.
    state_vars = ['counts', 'words']

    def __init__(self, **params):
        raise NotImplementedError("BaseLM is an abstract class; do not use directly.")

    def __eq__(self, other):
        """For testing. Check if two models are equal."""
        return all([getattr(self, v) == getattr(other, v) for v in self.state_vars])

    def set_live_params(self, **params):
        """Set parameters of the model to be used for future predictions."""
        raise NotImplementedError("set_live_params() must be implemented by subclass.")

    def next_word_proba(self, word, context, **kw):
        """Predict the probability of a word given the preceding words
        (context)."""
        raise NotImplementedError("next_word_proba() must be implemented by subclass.")

    def sample_next(self, context, **kw):
        """Sample a word from the conditional distribution.

        Args:
            lm: AddKTrigramLM or KNTrigramLM
            context: list(string) representing previous words
              (e.g. [w_0, ..., w_i-2, w_i-1])
            **kw: additional keywords, passed to self.next_word_proba

        Returns:
            (string) the next word to predict
        """
        probs = [self.next_word_proba(word, context, **kw) for word in self.words]
        return np.random.choice(self.words, p=probs)
    
    def score_seq(self, seq, verbose=False):
        """Compute log probability (base 2) of the given sequence.

        Args:
            seq: sequence of words (tokens) to score
            verbose: if true, will print per-token probabilities

        Returns:
            (score, num_real_tokens)
            score: log-likelihood of sequence
            num_real_tokens: number of tokens scored, excluding <s> and </s>
        """
        context_size = self.order_n - 1
        score = 0.0
        count = 0
        # Start at third word, since we need a full context.
        for i in range(context_size, len(seq)):
            if (seq[i] == u"<s>" or seq[i] == u"</s>"):
                continue  # Don't count special tokens in score.
            context = seq[i-context_size:i]
            s = np.log2(self.next_word_proba(seq[i], context))
            score += s
            count += 1
            # DEBUG.
            if verbose:
                print("log P({:s} | {:s}) = {:.03f}".format(seq[i], " ".join(context), s))
        return score, count

    def print_stats(self):
        """Output summary statistics about our language model."""
        print("=== N-gram Language Model stats ===")
        for i in range(self.order_n):
            unique_ngrams = sum(len(c) for k,c in self.counts.items()
                    if len(k) == i)
            print("{:8,} unique {:d}-grams".format(unique_ngrams, i+1))

        optimal_memory_bytes = sum(
                (4 * len(k) + 20 * len(v))
                 for k, v in self.counts.items())
        print("Optimal memory usage (counts only): {:.02f} MB".format(optimal_memory_bytes / (2**20)))


class AddKTrigramLM(BaseLM):
    """Trigram LM with add-k smoothing."""
    order_n = 3
    # For testing - do not modify.
    state_vars = ['k', 'counts', 'context_totals', 'words', 'V']

    def __init__(self, tokens):
        """Build our smoothed trigram model.

        This should be very similar to SimpleTrigramLM.__init__ from the demo
        notebook, with the exception that we _don't_ want to actually normalize
        the probabilities at training time. Instead, we'll compute the corpus
        counts C_abc = C(w_2, w_1, w) and C_ab = C(w_2, w_1), after which we can
        compute the probabilities on the fly for any value of k. (We'll do this
        in the next_word_proba() function.)

        The starter code will fill in:
          self.counts  (trigram counts)
          self.words   (list of words known to the model)

        Your code should populate:
          self.context_totals (total count C_ab for context ab)

        Args:
          tokens: (list or np.array) of training tokens

        Returns:
          None
        """
        self.k = 0.0
        # Raw trigram counts over the corpus.
        # c(w | w_1 w_2) = self.counts[(w_2,w_1)][w]
        # Be sure to use tuples (w_2,w_1) as keys, *not* lists [w_2,w_1]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Map of (w_1, w_2) -> int
        # Entries are c( w_2, w_1 ) = sum_w c(w_2, w_1, w)
        self.context_totals = defaultdict(lambda: 0.0)

        # Track unique words seen, for normalization
        # Use wordset.add(word) to add words
        wordset = set()

        # Iterate through the word stream once
        # Compute trigram counts as in SimpleTrigramLM
        w_1, w_2 = None, None
        for word in tokens:
            wordset.add(word)
            if w_1 is not None and w_2 is not None:
                self.counts[(w_2,w_1)][word] += 1
            # Update context
            w_2 = w_1
            w_1 = word

        #### YOUR CODE HERE ####
        # Compute context counts
        for context, words in self.counts.items():
            self.context_totals[context] = sum(words[w] for w in words)

        #### END(YOUR CODE) ####
        # Freeze defaultdicts so we don't accidentally modify later.
        self.counts.default_factory = None
        for k in self.counts:
            if isinstance(self.counts[k], defaultdict):
                self.counts[k].default_factory = None

        # Total vocabulary size, for normalization
        self.words = list(wordset)
        self.V = len(self.words)

    def set_live_params(self, k=0.0, **params):
        self.k = k

    def next_word_proba(self, word, seq):
        """Next word probability for smoothed n-gram.

        Your code should implement the corresponding equation from the
        notebook, using self.counts and self.context_totals as defined in
        __init__(), above.

        Be sure that you don't modify the parameters of the model in this
        function - in particular, you shouldn't (intentionally or otherwise)
        insert zeros or empty dicts when you encounter an unknown word or
        context. See note on dict.get() below.

        Args:
          word: (string) w in P(w | w_1 w_2 )
          seq: list(string) [w_1, w_2, w_3, ...]

        Returns:
          (float) P_k(w | w_1 w_2), according to the model
        """
        context = tuple(seq[-2:])  # (w_2, w_1)
        k = self.k
        
        #### YOUR CODE HERE ####
        # Hint: self.counts.get(...) and self.context_totals.get(...) may be
        # useful here. See note in dict_notes.md about how this works.
        V = self.V
        d = self.counts.get(context, {}) 
        return (d.get(word,0) + k) / (self.context_totals.get(context,0) + k * V)

        #### END(YOUR CODE) ####



class KNTrigramLM(BaseLM):
    """Trigram LM with Kneser-Ney smoothing."""
    order_n = 3
    # For testing - do not modify.
    state_vars = ['delta', 'counts', 'type_contexts',
                  'context_totals', 'context_nnz', 'type_fertility',
                  'z_tf', 'words']

    def __init__(self, tokens):
        """Build our smoothed trigram model.

        This should be similar to the AddKTrigramLM.__init__ function, above,
        but will compute a number of additional quantities that we need for the
        more sophisticated KN model.

        See the documentation in the notebook for the KN backoff model
        definition and equations, and be sure to read the in-line comments
        carefully to understand what each data structure represents.

        Note the usual identification of variables:
          w : c : current word
          w_1 : w_{i-1} : b : previous word
          w_2 : w_{i-2} : a : previous-previous word

        There are two blocks of code to fill here. In the first one, you should
        fill in the inner loop to compute:
          self.counts         (unigram, bigram, and trigram)
          self.type_contexts  (set of preceding words for each word (type))

        In the second one, you should compute:
          self.context_totals  (as in AddKTrigramLM)
          self.context_nnz     (number of nonzero elements for each context)
          self.type_fertility  (number of unique preceding words for each word
                                      (type))

        The starter code will fill in:
          self.z_tf   (normalization constant for type fertilities)
          self.words  (list of words known to the model)

        Args:
          tokens: (list or np.array) of training tokens

        Returns:
          None
        """
        self.delta = 0.75
        # Raw counts over the corpus.
        # Keys are context (N-1)-grams, values are dicts of word -> count.
        # You can access C(w | w_{i-1}, ...) as:
        # unigram: self.counts[()][w]
        # bigram:  self.counts[(w_1,)][w]
        # trigram: self.counts[(w_2,w_1)][w]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0))
        # As in AddKTrigramLM, but also store the unigram and bigram counts
        # self.context_totals[()] = (total word count)
        # self.context_totals[(w_1,)] = c(w_1)
        # self.context_totals[(w_2, w_1)] = c(w_2, w_1)
        self.context_totals = defaultdict(lambda: 0.0)
        # Also store in self.context_nnz the number of nonzero entries for each
        # context; as long as \delta < 1 this is equal to nnz(context) as
        # defined in the notebook.
        self.context_nnz = defaultdict(lambda: 0.0)

        # Context types: store the set of preceding words for each word
        # map word -> {preceding_types}
        self.type_contexts = defaultdict(lambda: set())
        # Type fertility is the size of the set above
        # map word -> |preceding_types|
        self.type_fertility = defaultdict(lambda: 0.0)
        # z_tf is the sum of type fertilities
        self.z_tf = 0.0


        # Iterate through the word stream once
        # Compute unigram, bigram, trigram counts and type fertilities
        w_1, w_2 = None, None
        for word in tokens:
            
            #### YOUR CODE HERE ####
            
            # Unigram counts
            self.counts[()][word] += 1
            
            if w_1 is not None:
                # Bigram counts
                self.counts[(w_1,)][word] += 1
                
                # Unique context words for each word
                self.type_contexts[word].add(w_1)
                
                if w_2 is not None:
                    # Trigram counts
                    self.counts[(w_2,w_1)][word] += 1
            
            #### END(YOUR CODE) ####
            
            # Update context
            w_2 = w_1
            w_1 = word
            
        ##
        # We'll compute type fertilities and normalization constants now,
        # but not actually store the normalized probabilities. That way, we can compute
        # them (efficiently) on the fly.

        #### YOUR CODE HERE ####
        # Count the total for each context.
        for context, words in self.counts.items():
            self.context_totals[context] = sum(words[w] for w in words)
        
        # Count the number of nonzero entries for each context.
            for word, cnt in words.items():
                if cnt > self.delta:
                    self.context_nnz[context] += 1

        # Compute type fertilities, and the sum z_tf.
        for word, context in self.type_contexts.items():
            self.type_fertility[word] = len(context)
            
        self.z_tf = float(sum(self.type_fertility.values()))
        #### END(YOUR CODE) ####


        # Freeze defaultdicts so we don't accidentally modify later.
        self.counts.default_factory = None
        self.type_contexts.default_factory = None

        # Total vocabulary size, for normalization
        self.words = list(self.counts[()].keys())
        self.V = len(self.words)


    def set_live_params(self, delta = 0.75, **params):
        self.delta = delta

    def kn_interp(self, word, context, delta, pw):
        """Compute KN estimate P_kn(w | context) given a backoff probability

        Your code should implement the absolute discounting equation from the
        notebook, using the counts computed in __init__(). Note that you don't
        need to deal with type fertilities here; this is handled in the
        next_word_proba() function in the starter code, below.

        Be sure you correctly handle the case where c(context) = 0, so as to not
        divide by zero later on. You should just return the backoff probability
        directly, since we have no information to decide otherwise.

        Be sure that you don't modify the parameters of the model in this
        function - in particular, you shouldn't (intentionally or otherwise)
        insert zeros or empty dicts when you encounter an unknown word or
        context. See note on dict.get() below.

        Args:
          word: (string) w in P(w | context )
          context: (tuple of string)
          delta: (float) discounting term
          pw: (float) backoff P_kn(w | less_context), precomputed

        Returns:
          (float) P_kn(w | context)
        """
        pass
        #### YOUR CODE HERE ####
        
        # Hint: self.counts.get(...) and self.context_totals.get(...) may be
        # useful here. See note in dict_notes.md about how this works.
        d = self.counts.get(context, {})
        Cabc = d.get(word,0)
        Cab = self.context_totals.get(context,0)
        nnzb = self.context_nnz.get(context,0)
        
        if Cab == 0:
            return pw
        else:
            backoff = (delta / Cab) * nnzb
            if Cabc - delta < 0:
                return backoff * pw
            else:
                return ((Cabc - delta) / Cab) + (backoff * pw)

        #### END(YOUR CODE) ####


    def next_word_proba(self, word, seq):
        """Compute next word probability with KN backoff smoothing.

        Args:
          word: (string) w in P(w | w_1 w_2 )
          seq: list(string) [w_1, w_2, w_3, ...]
          delta: (float) discounting term

        Returns:
          (float) P_kn(w | w_1 w_2)
        """
        delta = delta = self.delta
        # KN unigram, then recursively compute bigram, trigram
        pw1 = self.type_fertility.get(word, 0.0) / self.z_tf
        pw2 = self.kn_interp(word, tuple(seq[-1:]), delta, pw1)
        pw3 = self.kn_interp(word, tuple(seq[-2:]), delta, pw2)
        return pw3
