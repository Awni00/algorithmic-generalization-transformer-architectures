class FactoredVocabTokenizer():
    """
    A tokenizer class for handling factored vocabularies in a computational graph context.

    Vocabulary has a factored structure with the following factors:
    - Syntax: VARIABLE, OPERATION, VALUE, EQUAL, EQ_SEP
    - Variable: x0, x1, ..., x_{n-1}, if token is variable
    - Operation: ADD, MUL, ..., if token is operation
    - Value: 0, 1, ..., mod_val - 1, if token is leaf value or variable; N/A if token is not value; EMPTY if variable is not computed yet

    Attributes:
        n_vars (int): Number of variables.
        ops (list): List of operations.
        mod_val (int): Modulus value for the value partition.
        EQUAL (str): Equal sign token.
        EQ_SEP (str): Separator token for different factors in vocab.
        syntax (list): List of syntax types.
        value_partition (list): List of possible values.
        operation_partition (list): List of possible operations.
        variable_partition (list): List of possible variables.
        value_toks (list): List of value tokens including 'N/A' and 'EMPTY'.
        operation_toks (list): List of operation tokens including 'N/A'.
        variable_toks (list): List of variable tokens including 'N/A'.
        syntax_idx2tok (dict): Mapping from syntax index to token.
        syntax_tok2idx (dict): Mapping from syntax token to index.
        value_idx2tok (dict): Mapping from value index to token.
        value_tok2idx (dict): Mapping from value token to index.
        operation_idx2tok (dict): Mapping from operation index to token.
        operation_tok2idx (dict): Mapping from operation token to index.
        variable_idx2tok (dict): Mapping from variable index to token.
        variable_tok2idx (dict): Mapping from variable token to index.
    Methods:
        get_token_syntax(token): Returns the syntax type of a given token.
        factor_token(token): Factors a single token into its components.
        factor_tokens(list_tokens): Factors a list of tokens into their components.
        factor_string(string, sep=' '): Factors a string of tokens into their components.
        encode_token(token): Encodes a single token into its indices.
        decode_token(token): Decodes a single token from its indices.
        encode_tokens(list_tokens): Encodes a list of tokens into their indices.
        decode_tokens(list_indices): Decodes a list of tokens from their indices.
        encode_string(string, sep=' '): Encodes a string of tokens into their indices.
        decode_string(list_indices, sep=' '): Decodes a string of tokens from their indices.
    """

    def __init__(self, n_vars, ops, mod_val, max_fan_in):
        self.n_vars = n_vars
        self.ops = ops
        self.mod_val = mod_val
        self.max_fan_in = max_fan_in

        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        # self.eos_token = '<EOS>'
        self.EQUAL = '='
        self.EQ_SEP = '<EQ_SEP>'

        # add additional special tokens that will be used in the 'scratch pad' in the intermediate steps
        # NOTE: we add two factors for REGRET and MODIFICATION
        self.TRUE = 'TRUE'
        self.FALSE = 'FALSE'

        self.special_tokens = [self.PAD, self.UNK, self.EQUAL, self.EQ_SEP]

        self.syntax = ['VARIABLE', 'OPERATION', 'VALUE', *self.special_tokens] # possible syntax types

        self.value_partition = [str(val) for val in range(mod_val)] # values are from 0 to mod_val - 1
        self.operation_partition = ops # operations are from ops
        self.variable_partition = [f'x_{i}' for i in range(n_vars)] # variables are from x0 to x{n-1}

        self.value_toks = self.value_partition + ['N/A', 'EMPTY']
        self.operation_toks = self.operation_partition + ['N/A'] # N/A for tokens that are not operations (e.g., value, variable)
        self.variable_toks = self.variable_partition + ['N/A']

        self.syntax_idx2tok = {i: tok for i, tok in enumerate(self.syntax)}
        self.syntax_tok2idx = {tok: i for i, tok in enumerate(self.syntax)}

        self.value_idx2tok = {i: tok for i, tok in enumerate(self.value_toks)}
        self.value_tok2idx = {tok: i for i, tok in enumerate(self.value_toks)}

        self.operation_idx2tok = {i: tok for i, tok in enumerate(self.operation_toks)}
        self.operation_tok2idx = {tok: i for i, tok in enumerate(self.operation_toks)}

        self.variable_idx2tok = {i: tok for i, tok in enumerate(self.variable_toks)}
        self.variable_tok2idx = {tok: i for i, tok in enumerate(self.variable_toks)}

        self.regret_idx2tok = {i: tok for i, tok in enumerate([self.FALSE, self.TRUE])}
        self.regret_tok2idx = {tok: i for i, tok in enumerate([self.FALSE, self.TRUE])}

        self.modification_idx2tok = {i: tok for i, tok in enumerate([self.FALSE, self.TRUE])}
        self.modification_tok2idx = {tok: i for i, tok in enumerate([self.FALSE, self.TRUE])}

        self.vocab_sizes = (len(self.syntax), len(self.variable_toks), len(self.operation_toks), len(self.value_toks), len(self.regret_idx2tok), len(self.regret_idx2tok))
        self.factors = ['SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE', 'REGRET', 'MODIFICATION']
        self.n_factors = len(self.vocab_sizes)

    def get_token_syntax(self, token):
        # e.g., x0 is mapped to VARIABLE, ADD is mapped to OPERATION, 0 is mapped to VALUE
        if token in self.special_tokens:
            return token
        elif token in self.variable_partition:
            return 'VARIABLE'
        elif token in self.operation_partition:
            return 'OPERATION'
        elif token in self.value_partition:
            return 'VALUE'
        else:
            return self.UNK

    def factor_token(self, token):
        # e.g., x0 is mapped to ('VARIABLE', 'x0', 'N/A', 'EMPTY')
        # e.g., ADD is mapped to ('OPERATION', 'N/A', 'ADD', 'N/A')
        # e.g., 0 is mapped to ('VALUE', 'N/A', 'N/A', '0')

        syntax = self.get_token_syntax(token)

        if syntax == 'VARIABLE':
            variable = token
            operation = 'N/A'
            value = 'EMPTY' # value is not computed yet
        elif syntax == 'OPERATION':
            variable = 'N/A'
            operation = token
            value = 'N/A'
        elif syntax == 'VALUE':
            variable = 'N/A'
            operation = 'N/A'
            value = token
        else:
            variable = 'N/A'
            operation = 'N/A'
            value = 'N/A'

        # return [syntax, variable, operation, value]
        # NOTE: We additionally add two factors for REGRET and MODIFICATION. Set to FALSE by default.
        return [syntax, variable, operation, value, self.FALSE, self.FALSE]

    def factor_tokens(self, list_tokens):
        return [self.factor_token(tok) for tok in list_tokens]

    def factor_string(self, string, sep=' '):
        return self.factor_tokens(string.split(sep))

    def de_factor_token(self, factors):
        # given factors, returns the original token
        # e.g., ('VARIABLE', 'x0', 'N/A', 'EMPTY') is mapped to x0

        syntax, variable, operation, value, regret, modification = factors

        if syntax == 'VARIABLE':
            return variable
        elif syntax == 'OPERATION':
            return operation
        elif syntax == 'VALUE':
            return value
        else:
            return syntax

    def de_factor_tokens(self, list_factors):
        return [self.de_factor_token(factors) for factors in list_factors]

    def encode_token(self, token):
        # returns idx tokens of syntax, variable, operation, value (see factor_token)

        factor_ls = self.factor_token(token)

        # NOTE: default factors are ['SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE', 'REGRET', 'MODIFICATION']
        syntax, variable, operation, value, regret, mod = factor_ls

        syntax_idx = self.syntax_tok2idx[syntax]
        variable_idx = self.variable_tok2idx[variable]
        operation_idx = self.operation_tok2idx[operation]
        value_idx = self.value_tok2idx[value]

        # NOTE: regret and mod are not used in the model
        regret_idx = self.regret_tok2idx[regret]
        mod_idx = self.modification_tok2idx[mod]

        return [syntax_idx, variable_idx, operation_idx, value_idx, regret_idx, mod_idx]

    def encode_tokens(self, list_tokens):
        return [self.encode_token(tok) for tok in list_tokens]

    def encode_factored_token(self, factors):
        # given factors, returns idx tokens of syntax, variable, operation, value

        syntax, variable, operation, value = factors[0:4]
        regret, mod = factors[4:6]

        syntax_idx = self.syntax_tok2idx[syntax]
        variable_idx = self.variable_tok2idx[variable]
        operation_idx = self.operation_tok2idx[operation]
        value_idx = self.value_tok2idx[value]

        regret_idx = self.regret_tok2idx[regret]
        mod_idx = self.modification_tok2idx[mod]

        return [syntax_idx, variable_idx, operation_idx, value_idx, regret_idx, mod_idx]

    def encode_factored_tokens(self, list_factors):
        return [self.encode_factored_token(factors) for factors in list_factors]

    def decode_token(self, token):
        # given idx tokens of syntax, variable, operation, value, returns the original token factors

        syntax_idx, variable_idx, operation_idx, value_idx, regret_idx, mod_idx = token
        syntax = self.syntax_idx2tok[syntax_idx]
        variable = self.variable_idx2tok[variable_idx]
        operation = self.operation_idx2tok[operation_idx]
        value = self.value_idx2tok[value_idx]

        regret = self.regret_idx2tok[regret_idx]
        mod = self.modification_idx2tok[mod_idx]

        return syntax, variable, operation, value, regret, mod

    def decode_tokens(self, list_tokens):
        return [self.decode_token(idx) for idx in list_tokens]

    def encode_string(self, string, sep=' '):
        return self.encode_tokens(string.split(sep))

    def decode_string(self, list_indices, sep=' '):
        return sep.join([self.decode_token(idx) for idx in list_indices])

class CoTTokenizer():

    def __init__(self, n_vars, ops, mod_val):
        self.n_vars = n_vars
        self.ops = ops
        self.mod_val = mod_val

        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        # self.eos_token = '<EOS>'
        self.EQUAL = '='
        self.EQ_SEP = '<EQ_SEP>'
        self.COT = '<COT>'

        self.special_tokens = [self.PAD, self.UNK, self.EQUAL, self.EQ_SEP, self.COT]

        self.value_partition = [str(val) for val in range(mod_val)] # values are from 0 to mod_val - 1
        self.operation_partition = ops # operations are from ops
        self.variable_partition = [f'x_{i}' for i in range(n_vars)] # variables are from x0 to x{n-1}

        self.vocab = self.special_tokens + self.value_partition + self.operation_partition + self.variable_partition

        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}

    def encode_tokens(self, tokens):
        # given a list of tokens, encode it into a list of integer
        return [self.token2idx[token] for token in tokens]

    def encode_string(self, string):
        # given a string, encode it into a list of integer tokens
        return [self.token2idx[token] for token in string.split()]

    def decode_tokens(self, token_list, return_string=False):
        # given a list of integer tokens, decode it into a string
        tokens = [self.idx2token[token] for token in token_list]
        if return_string:
            return ' '.join(tokens)
        else:
            return tokens

    def decode_string(self, token_list, sep=' '):
        # given a list of integer tokens, decode it into a string
        return sep.join([self.idx2token[token] for token in token_list])