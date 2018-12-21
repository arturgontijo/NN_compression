import numpy as np
import argparse
import json


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='iid',
                        help='the type of data that needs to be generated')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='length of the sequence to be generated')
    parser.add_argument('--markovity', type=int, default=30,
                        help='Step for Markovity')
    parser.add_argument('--file_name', type=str, default='input.txt',
                        help='The name of the output file')
    parser.add_argument('--info_file', type=str, default='input_info.txt',
                        help='Name of the info file')
    parser.add_argument('--p1', type=float, default=0.5,
                        help='the probability for the entire sequence, or the base')
    parser.add_argument('--n1', type=float, default=0.0,
                        help='the probability for the entire sequence, or the base')
    
    return parser


# Computes the binary entropy
def entropy_iid(prob):
    p1 = prob
    p0 = 1.0 - prob
    h = -(p1*np.log(p1) + p0*np.log(p0))
    h /= np.log(2.0)
    return h


def main():
    parser = get_argument_parser()
    flags = parser.parse_args()
    flags.p0 = 1.0 - flags.p1
    flags.n0 = 1.0 - flags.n1
    _keys = ["data_type", "p1", "n1"]

    data = np.empty([flags.num_samples, 1], dtype='S1')
    # print data.shape
    
    if flags.data_type == 'iid':
        # Generate data
        data = np.random.choice(['a', 'b'], size=(flags.num_samples, 1), p=[flags.p0, flags.p1])
        flags.Entropy = entropy_iid(flags.p1)
        _keys.append("Entropy")
 
    elif flags.data_type == '0entropy':
        data[:flags.markovity, :] = np.random.choice(['a', 'b'], size=(flags.markovity, 1), p=[flags.p0, flags.p1])
        for i in range(flags.markovity, flags.num_samples):
            if data[i-1] == data[i-flags.markovity]:
                data[i] = 'a'
            else:
                data[i] = 'b'
        flags.Entropy = 0
        _keys.append("Entropy")
        _keys.append("markovity")
    
    elif flags.data_type == 'HMM':
        data[:flags.markovity, :] = np.random.choice(['a', 'b'], size=(flags.markovity, 1), p=[flags.p0, flags.p1])
        for i in range(flags.markovity, flags.num_samples):
            if data[i-1] == data[i-flags.markovity]:
                data[i] = np.random.choice(['a', 'b'], p=[flags.n0, flags.n1])
            else:
                data[i] = np.random.choice(['b', 'a'], p=[flags.n0, flags.n1])
  
        flags.Entropy = entropy_iid(flags.n1) 
        _keys.append("Entropy")
        _keys.append("markovity")
        print("HMM Data generated ...")

    np.savetxt(flags.file_name, data, delimiter='', fmt='%c', newline='')
    
    # print _keys
    args = vars(flags)
    info = {key: args[key] for key in _keys}
    # print info
    with open(flags.info_file, "wb") as f:
        json.dump(info, f)


if __name__ == '__main__':
    main()
