import numpy as np
from random import sample
import os
import pickle


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)


def remover(x): return ''.join([i for i in x if not i.isdigit()])


def get_celeb_dict(path):
    with open(path, 'rb') as f:
        data = f.read()
    data = data.split('\n')
    while '' in data:
        data.remove('')
    all_names = {}
    for i in xrange(len(data)):
        all_names[data[i]] = i
    return all_names


def get_name(name):
    return name[:-1].replace('_', ' ')


def simple_save(path, name, celeb_dict):
    HOME = os.path.expanduser('~')
    with open(HOME + path, 'rb') as f:
        faces = pickle.load(f)
    face_labels = map(lambda x: celeb_dict[get_name(x[1])], faces)
    face_images = map(lambda x: x[0], faces)
    size = len(face_labels)
    perm = sample(xrange(size), size)
    tr_size = int(0.9 * size)
    train_idxs = perm[:tr_size]
    test_idxs = perm[tr_size:]
    train = {'X': np.asarray(face_images)[train_idxs],
             'y': np.asarray(face_labels)[train_idxs]}
    test = {'X': np.asarray(face_images)[test_idxs],
            'y': np.asarray(face_labels)[test_idxs]}
    save_pickle(train, name + '/train.pkl')
    save_pickle(test, name + '/test.pkl')


def get_reverse_dict(sample_dict):
    rev = {}
    for key in sample_dict:
        val = sample_dict[key]
        rev[val] = key
    return rev


def labeled_save(real_path, caric_path, name, celeb_dict):
    HOME = os.path.expanduser('~')
    with open(HOME + real_path, 'rb') as f:
        real_faces = pickle.load(f)
    with open(HOME + caric_path, 'rb') as f:
        caric_faces = pickle.load(f)

    label_dict = {}
    # initializing each class of label_dict
    for i in xrange(len(celeb_dict)):
        label_dict[i] = {'real': [],
                         'caric': []}

    # collecting images per label
    for img, lbl in real_faces:
        try:
            lbl = celeb_dict[get_name(lbl)]
            label_dict[lbl]['real'] += [img]
        except:
            pass
    for img, lbl in caric_faces:
        lbl = celeb_dict[get_name(lbl)]
        label_dict[lbl]['caric'] += [img]

    rev = get_reverse_dict(celeb_dict)
    # converting to ndarrays
    for i in xrange(len(label_dict)):
        if len(label_dict[i]['caric']) > 0:
            label_dict[i]['real'] = np.asarray(label_dict[i]['real'])
            label_dict[i]['caric'] = np.asarray(label_dict[i]['caric'])
        else:
            print 'popping label', i, '- no caricature found for', rev[i]
            label_dict.pop(i)

    # saving the dict
    save_pickle(label_dict, name)


if __name__ == "__main__":
    HOME = os.path.expanduser('~')
    celeb_dict = get_celeb_dict(HOME + '/data/scripts/celeb_list.txt')
    simple_save(path='/data/reshaped_cropped_real_64x64/fullset.pkl',
                name='real-face',
                celeb_dict=celeb_dict)
    simple_save(path='/data/reshaped_cropped_64x64/fullset.pkl',
                name='caricature-face',
                celeb_dict=celeb_dict)
    labeled_save(real_path='/data/reshaped_cropped_real_64x64/fullset.pkl',
                 caric_path='/data/reshaped_cropped_64x64/fullset.pkl',
                 name='class-combined.pkl',
                 celeb_dict=celeb_dict)
