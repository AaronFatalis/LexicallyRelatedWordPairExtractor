from __future__ import unicode_literals

import nltk

from nltk.util import LazyMap
from nltk.corpus.reader.api import *


class MlrsReader(nltk.corpus.reader.conll.ConllCorpusReader):

    WORDS = 'words'   #: column type for words
    POS = 'pos'       #: column type for part-of-speech tags
    LEMMA = 'lemma'   #: column type for lemmatised words
    ROOT = 'root'     #: column type for root
    
    COLUMN_TYPES = (WORDS, POS, LEMMA, ROOT)
    
    def _read_grid_block(self, stream):
        """
        Modified _read_grid_block method - goes through files line by line
        and seperates blocks once '</s>' is found in the file,
        signifying the end of sentence. ~ is used in the reading process to
        to mark where a sentence ends.
        """
        
        s = ''

        while True:
            line = stream.readline()
            if not line:
                if s: break
                else: s = []
            elif line.strip() == '</s>':
                s += '~'
                if s: break
            else:
                s += line
                      
        for block in [s]:
            block = block.strip()

            if not block: continue

            pre_grid = []
            grid = []
            
            for line in block.split('\n'):
                splitline = line.split()
                try:
                    if splitline[0] == '~':
                        grid.append(pre_grid)
                        pre_grid = []
                    if len(splitline) == 4:
                        pre_grid.append(splitline)
                    
                except:
                    pass
            
        return grid

    def tagged_lem_sents(self, fileids=None, tagset=None):
        self._require(self.WORDS, self.POS, self.LEMMA)

        def get_tagged_lemmas(grid):
            return self._get_tagged_lemmas(grid, tagset)
        return LazyMap(get_tagged_lemmas, self._grids(fileids))

    def _get_tagged_lemmas(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        lemmas = self._get_column(grid, self._colmap['lemma'])
        words = self._get_column(grid, self._colmap['words'])
        lemwords = []
        
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]

        for l in lemmas:
            if l == 'null':
                ind = lemmas.index(l)
                word = words[ind]
                lemwords.append(word.lower())
            else:
                lemwords.append(l.lower())
                    
        return list(zip(lemwords, pos_tags))
    
if __name__ == "__main__":
    a = MlrsReader('../../../CorpusSplit', r'.*\.txt',["words","pos","lemma","root"])
