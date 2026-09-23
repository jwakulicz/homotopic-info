from argparse import ArgumentDefaultsHelpFormatter
from logging import root
import numpy as np
from numpy.linalg import LinAlgError
from vmmp_gmm.h_signature import HSignature

def get_suffix(h_sig):
    '''Returns longest suffix of an input HSignature as a HSignature
    
    Inputs
    -------
    h_sig : HSignature object
        HSignature object to get longest suffix of
        
    Returns
    --------
    HSignature object
    '''
    if len(h_sig.sig) == 1:
        return HSignature(())
    else:
        return HSignature(h_sig.sig[-(len(h_sig.sig)-1):])

def get_prefix(h_sig):
    if isinstance(h_sig, (HSignature)):
        h_sig = h_sig.sig
    if len(h_sig) == 1:
        return HSignature(())
    else:
        return HSignature(h_sig[:(len(h_sig)-1)])
    
def get_all_prefixes(h_sig):
    if isinstance(h_sig, (HSignature)):
        h_sig = h_sig.sig
    if len(h_sig) == 1:
        return [HSignature(())]
    else:
        sigs = [HSignature(h_sig[:i]) for i in range(len(h_sig))]
        return sigs
    
def is_prefix(h_sig, subtup):
    if subtup.sig in [sig.sig for sig in get_all_prefixes(h_sig)]:
        return True
    else:
        return False
    
def is_subtup(tup, subtup):
    return any(subtup == tup[i:i+len(subtup)] for i in range(0,len(tup)))

def is_strict_subtup(tup, subtup):
    return any(subtup == tup[0:i] for i in range(0,len(subtup)+1))

def build_til_sfx(sub_alpha, alphabet, sfx_list, wrd_len):
    sigs = get_all_strings(sub_alpha, wrd_len-1)
    missing_sigs = [sig.sig + l for sig in sigs for l in alphabet]
    prune_sigs = [
                  sig for sig in missing_sigs
                  if not any(is_subtup(sig[:-1], state) for state in sfx_list)
                 ]
    final_sigs = [
                  sig for sig in prune_sigs
                  if any(x in sfx_list for x in get_all_suffixes_tuple(sig))
                 ]
    final_sigs = final_sigs + sub_alpha + [()]
    return final_sigs

def get_all_suffixes(h_sig, tup_flag=False):
    ''' Returns a list of HSignature suffixes of a given HSignature
    
    Inputs
    ---------
    h_sig : HSignature object
        HSignature object to extract all suffixes from.
        
    Returns
    ---------
    list of HSignature objects
    '''
    if isinstance(h_sig, (tuple)): 
        s = h_sig
    elif isinstance(h_sig, (HSignature)):
        s = h_sig.sig

    if s == ('',) or s == ():
        if tup_flag:
            return ()
        else:
            return [HSignature(())]
    else:
        if tup_flag:
            return [s[-i:] for i in range(1, len(s)+1)]
        else:
        # print([h_sig.sig[-i:] for i in range(1, len(h_sig.sig)+1)])
            return [HSignature(s[-i:]) for i in range(1, len(s)+1)]

def get_all_suffixes_tuple(h_sig_tuple):
    if h_sig_tuple == ('',) or h_sig_tuple == ():
        return ()
    else:
        return [h_sig_tuple[-i:] for i in range(1, len(h_sig_tuple)+1)]

def count_subseq(subseq, seq):
    ''' Counts how many times a HSignature subsequence occurs in a longer HSignature.
    
    Inputs
    -------
    subseq : HSignature 
    seq : HSignature
    '''
    subseq_tup = subseq.sig
    seq_tup = seq.sig
    return sum(subseq_tup == seq_tup[i:len(subseq_tup) + i] for i in range(len(seq_tup) - len(subseq_tup) +1))

def get_all_strings(alphabet, wrd_len, pfx=HSignature(()), pfx_list=None):
    ''' Return a list of all possible words with specified length comprising characters
    from specified alphabet.
    
    Inputs
    --------
    alphabet : list (or some iterable container)
        list of tuples of characters that all data is constructed from
    wrd_len : int
        desired length of words in integer form
    
    Returns
    --------
    pfx_list : list of HSignatures.
        all possible words of desired length constructed from alphabet. '''
    if pfx_list == None:
        pfx_list = []

    if wrd_len == 0 or wrd_len == -1:
        pfx_list.append(pfx)
        return pfx_list

    for i in range(len(alphabet)):
        new_pfx = pfx.sig + alphabet[i]
        get_all_strings(alphabet, wrd_len - 1, HSignature(new_pfx), pfx_list)
    
    else:
        return pfx_list


def get_string_freq(string, obs, is_root=False):
    ''' Returns the frequency a sub-string appears in the data
    
    Inputs
    --------
    string : HSignature object containing sub-HSignature 
        the HSignature substring whose frequency is being calculated
    obs: list (or some iterable container)
        list/container of observed HSignatures
    
    Returns
    ---------
    count: int
        number of times the substring appears in data'''

    count = 0
    
    if is_root:
        for hsig in obs:
            if hsig.sig == string.sig:
                count += 1
    else:
        for hsig in obs:
            count += count_subseq(string, hsig)
    return count

def get_string_length_freq(string, obs, alphabet):
    ''' Returns the frequency any sub-string of a particular length constructed from
     the alphabet occurs in the data.
     
    Inputs
    --------
    string : HSignature 
        the HSignature sub-string whose length is being compared
    obs : list (or some other iterable container)
        list/container of observed HSignatures
    alphabet : list (or some iterable container)
        list of tuples of characters that all data is constructed from
    
    Returns
    --------
    count : int
        number of times any sub-string of len(string) appears in data'''
    count = 0
    string_len = len(string.sig)

    if string_len == 0:
        count += get_string_freq(HSignature(()), obs)
        return count

    else:
        generated_set = get_all_strings(alphabet, string_len)

        for gen_hsig in generated_set:
            count += get_string_freq(gen_hsig, obs)
        return count

def get_trns_prob_dict(node, alphabet, obs):
    ''' Returns probabilities of observing each letter in the alphabet after a string
    
    Inputs
    --------
    node : tuple
        the HSignature.sig tuple identifying the node
    alphabet : list (or some iterable container)
        list of tuples of characters that all data is constructed from
    obs : list (or some other iterable container)
        list/container of observed HSignatures

    Returns
    ---------
    trns_prob : dictionary
        dictionary of transition probabilities. key is the letter of the alphabet as a tuple, value
        is the probability of observing that letter after the node in data.
    '''
    trns_prob = {}
    for letter in alphabet:
        prob = calc_transition_prob(letter, node, obs, alphabet)
        trns_prob[letter] = prob
    return trns_prob

def calc_string_prob(h_sig, obs, alphabet):
    ''' Returns the probability of a sub-string occuring in data as according to
    Laplace's rule of succession

    Inputs
    --------
    h_sig : HSignature object
        HSignature of the sub-string whose probability is being calculated
    alphabet : list (or some iterable container)
        strings of characters that all data is constructed from
    obs : list (or some other iterable container)
        list/container of observed HSignatures

    Returns
    --------
    prob : float
        probability of input string occuring in data
    '''
    freq_s = get_string_freq(h_sig, obs)
    all_s = get_string_length_freq(h_sig, obs, alphabet)

    prob = (freq_s + 1) / (all_s + len(alphabet))
    return prob

def calc_transition_prob(sfx, h_sig, obs, alphabet):
    ''' Calculate the probability that sfx appears after string in data 
    
    Inputs
    -------- 
    sfx : string
        the single character suffix, a letter from the alphabet
    h_sig : HSignature
        the HSignature of the sub-string appearing before suffix in question, usually
        HSignature of a node in the tree
    alphabet : list (or some iterable container)
        strings of characters that all data is constructed from
    obs : list (or some other iterable container)
        list/container of observed HSignatures
    
    Returns
    --------
    prob : float
        probability that the given suffix appears after the given string in data'''
    if h_sig.sig == ():
        root_flag = True
    else:
        root_flag = False

    new_sig = HSignature(h_sig.sig + sfx)
    freq_sfx = get_string_freq(new_sig, obs, root_flag)
    sig_list = [HSignature(h_sig.sig + alpha_sfx) for alpha_sfx in alphabet]
    freq_all_sfx = 0

    for s in sig_list: 
        freq_all_sfx += get_string_freq(s, obs, root_flag)
    
    prob = (freq_sfx + 1) / (freq_all_sfx + len(alphabet))
    return prob

def err_fn(child, parent, alphabet, obs):
    ''' Calculate the weighted KL divergence error function from Power of Amnesia paper. 
    Compares how significant a child suffix is against it's parent.
    
    Inputs
    -------
    child : HSignature
        HSignature child of parent input, a new suffix s = s2s1 built on parent s1.
    parent : HSignature
        HSignature of parent suffix s
    alphabet : list (or some iterable container)
        strings of characters that all data is constructed from
    obs : list (or some other iterable container)
        list/container of observed HSignatures
    
    Returns
    --------
    err: float
        result of error function'''
    prob_child = calc_string_prob(child, obs, alphabet)

    err = 0
    for sfx in alphabet:
        prob_follow_child = calc_transition_prob(sfx, child, obs, alphabet)
        prob_follow_parent = calc_transition_prob(sfx, parent, obs, alphabet)
        err += prob_follow_child * (np.log(prob_follow_child) - np.log(prob_follow_parent))
    err = prob_child * err
    return err

def get_candidate_set(S, sfx, obs, epsilon, alphabet):
    ''' Return the candidate set for generating a VMMP from data
    
    Inputs
    --------
    S : list
        the current list of suffixes being considered for addition to VMMP
    sfx : HSIgnature
        the current HSignature being added to the VMMP
    obs : list (or some other iterable container)
        list/container of observed strings
    epsilon : float
        occurance threshold, probability of the candidate string must be larger than
        this threshold to be added to the set
    alphabet : list (or some iterable container)
        strings of characters that all data is constructed from
    
    Returns
    --------
    S : list
        new S with added suffixes to be considered for addition to VMMP
    '''
    for letter in alphabet:
        new_sfx = HSignature(letter + sfx.sig)
        if calc_string_prob(new_sfx, obs, alphabet) >= epsilon:
            S.append(new_sfx)
    return S

if __name__ == '__main__':
    hsig = HSignature((1,2,3))
    print([pfx.sig for pfx in get_all_prefixes(hsig)])
    # tup = '12345'
    # subtup = '124'
    # print(is_strict_subtup(tup, subtup))
        # alph = [(1,), (-1,), (-2,), (2,)]
    # test = get_all_strings(alph, 1)
    # print([x.sig for x in test])

    # test1 = get_all_suffixes(HSignature(('1','0','0',)))
    # print([t.sig for t in test1])
    # test2 = get_prefix(HSignature(('1','0','0',)))
    # print(test2.sig)
    # for sig in test1:
    #     print(sig.sig)
    #     print(sig.sig + ('2',))

    # my_list = [('itemitem', 'item', 'a',)]
    # a = sum([el.count('item') for tup in my_list for el in tup])
    # print(a)

    # test_tup = ('1', '0', '-1', '1')
    # subset = ('-1','0')
    # subseq = ('0','-1',) 
    # subseq2 = ('1',)
    # # if set(('-1','0')).issubset(test_tup):
    # #     print('should not print, wrong order')                                       
    # # if set(('0','-1')).issubset(test_tup):
    # #     print('should print')
    # if any(subset == test_tup[i:len(subset) + i] for i in range(len(test_tup) - len(subset) +1)):
    #     print('should not print, wrong order')

    # if any(subseq == test_tup[i:len(subseq) + i] for i in range(len(test_tup) - len(subseq) +1)):
    #     print('should print')

    # test_sum = sum(subset == test_tup[i:len(subset) + i] for i in range(len(test_tup) - len(subset) +1))
    # test_sum_2 = sum(subseq == test_tup[i:len(subseq) + i] for i in range(len(test_tup) - len(subseq) +1))
    # test_sum_3 = sum(subseq2 == test_tup[i:len(subseq2) + i] for i in range(len(test_tup) - len(subseq2) +1))
    # print(test_sum, test_sum_2, test_sum_3)

    # print(subseq == test_tup[i:len(subseq) + i] for i in range(len(test_tup) - len(subseq) +1))

    # print(test_tup[0:1])
    # print(type(()))
    

