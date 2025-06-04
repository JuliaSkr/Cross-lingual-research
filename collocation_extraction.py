import os
import sys

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
os.chdir(script_directory)

# Теперь можно проверить, что рабочая директория изменена
print(f"Рабочая директория изменена на: {os.getcwd()}")

# IMPORT SECTION
import pandas as pd
import stanza

from tqdm import tqdm

# from datasets import load_dataset
# from huggingface_hub import login


# COLLOCATOR DEPENDENCIES
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import stanza
import pickle

from collections import Counter
import math
import json
import ast
import random
import os

from tqdm import tqdm
import multiprocessing

# CLASSES
class CollocatorDeluxe:
    def __init__(self,
                 model,
                 data=None,
                 cleaner=None,
                 sentence_trim=10,
                 text_trim=2,
                 sentence_sampling='start',
                 parse_deprel=False,
                 min_word_count=3,
                 min_bigram_count=3,
                 max_word_count=30,
                 max_bigram_count=30,
                 file_mode='pickle',
                 language_code='new',
                 max_csv_len = 10000,
                 output_directory=None):
        """
        # Коллокатор
        Класс Коллокатор для извлечения коллокаций из текстов.
        
        ## Параметры
        Ниже перечислены различные разделы параметров класса. Некоторые параметры указаны в нескольких разделах, так как при разном использовании имеют различный функционал.
        ### Основные
        Эти параметры определяют базовое поведение класса.

        - **model**: *stanza.Pipeline | None* - при инициализации модели станза, коллокатор становится 'обрабатывающим', тогда как без модели может только 'читать' файлы CSV и JSON, которые для него предназначены.
            
        - **data**: *dict | None = None* - при создании коллокатора можно сразу указать словарь данных. Если словарь не прошёл валидацию, он отклоняется классом.
            
        - **cleaner**: *Cleaner | None - Cleaner*, который будет использоваться при обработке текстов и валидации коллокаций на этапе просмотра и подсчёта статистик.

        ### Обработка
        Эти параметры используются во время обработки (чтения) текстовых файлов

        - **text_trim** : *int = 2* - определяет, сколько предложений из текста попадёт в коллокатор для обработки.
        - **sentence_trim**: *int = 10* - определяет, сколько слов из предложения попадёт в коллокатор для составления биграмм
        - **sentence_sampling**: *"start" | "random" = "start"* - определяет способ, которым коллокатор набирает предложения из текста (start - с начала текста, random - случайно)
        - **parse_deprel** *bool = False* - определяет, будут ли парситься зависимости между частями речи (часть речи - часть речи) при подборе биграмм

        ### Просмотр
        Эти параметры определяют, как будут фильтроваться и в каком виде просматриваться биграммы.

        - **min_bigram_count**: *int = 3* - определяет, сколько минимум раз должен повстречаться биграмм, чтобы считаться важным. При просмотре и фильтрации отметает все биграммы которые встретились меньше раз.
        - **max_bigram_count**: *int = 30* - определяет, сколько максимум раз может повстречаться биграмм прежде чем станет гиперчастотным. При просмотре и фильтрации отметает все биграммы, которые встретились больше раз.
        - **parse_deprel**: *bool = False* - определяет, будут ли показываться зависимости между биграммами (если есть)
        
        ### Сохранение и Загрузка
        Эти параметры определяют, как будут сохраняться файлы вывода для подсчитанных биграмм и статистик

        - **file_mode**: *"csv" | "json" = "csv"* - определяет, в каком формате класс будет сохранять и загружать подсчитанные слова и биграммы
        - **language_code**: *str  = "new"* - код языка, который будет использоваться при сохранении файлов в интервальных сохранениях
        - **max_csv_len**: *int = 10000* - ограничение на количество строк в сохраненных файлах CSV при интервальных сохранениях и сохранении счёта слов и биграмм. Используется для избежания слишком больших файлов CSV для экспорта.
        - **output_directory**: *str | None = None* - директория, которая будет использоваться для сохранения вывода класса. В качестве этой переменной используйте имя директории, которую хотите создать или использовать для хранения вывода.

        ## Использование
        Ниже будут представлены практические пути использования класса

        ### Создание класса

        Пример очень простого формирования коллокатора.

        ```
        from nltk.corpus import stopwords
        model = stanza.Pipeline('ru')
        coll = CollocatorDeluxe(model,
                                language_code='ru',
                                file_mode='csv',
                                output_directory='stats',
                                cleaner=Cleaner(stopwords=stopwords.words('russian),
                                                regex=[],
                                                filter_numbers=True,
                                                filter_punctuation=True,
                                                min_word_len=2))
        ```

        ### Обработка текста

        Загрузить текст в коллокатор для обработки можно, используя внутренний метод __call__:

        ```
        text = "Был холодный и ясный апрельский день и часы пробили тринадцать."
        coll(text)
        ```

        Во время длительной загрузки текстов коллокатор будет периодически делать интервальные сохранения. В зависимости от типа сохранения, в разной форме.

        **ВАЖНО**! Учтите, что обработка текста будет работать только в том случае, если определена модель stanza.Pipeline внутри коллокатора.

        #### Сохранение

        Сохранить данные коллокатора можно самостоятельно, используя `.save(name: str)`

        ```
        coll.save('save_name')
        # save_name указывается без расширения, потому что в зависимости от типа сохранения, он использует разные способы сохранения
        ```

        #### Загрузка

        В зависимости от режима, загрузка происходит разными способами.

        ##### CSV

        Для загрузки файлов CSV необходимо указать папку:

        ```
        coll.load('loading_directory')
        # Вместо loading_directory указать название своей папки, то есть, относительный путь к ней относительно рабочего пространства
        ```

        ##### JSON

        Для загрузки файла JSON необходимо указать относительный путь к файлу с его расширением:

        ```
        coll.load('data_file.json')
        # Указать своё название файла + путь к нему
        ```

        ### Просмотр и сохранение статистик

        Статистики считаются на этапе просмотра. Их можно представить в виде датафрейма. Датафрейм из коллокатора можно получить таким способом:

        ```
        coll.present_statistics()
        # Датафрейм будет получен с учётом очистки объектом класса Cleaner. Если по какой-то причине этот класс отсутствует, коллокатор обрабатывает исключения иным способом.
        ```

        Датафрейм также можно сохранить в CSV формате

        ```
        coll.save_statistics('save_directory')
        # Вместо save_directory указать своё название папки. При наличии self.output_directory save_directory будет помещён в общую директорию для вывода
        ```

        #### Леммы
        Внутри коллокатора существует также алгоритм обработки для лемм, то есть, лемматизация. Представить статистику для лемм можно похожим образом:

        ```
        coll.present_statistics_lemma()
        # На выходе будет датафрейм, но вместо обычных биграмм это будут леммы
        ```

        Сохранять так же:

        ```
        coll.save_statistics_lemma('save_directory')
        # На месте save_directory указать свою желаемую директорию
        ```

        Коллокатор также имеет

        ## Зависимости

        ```
        import pandas as pd
        import stanza
        from collections import Counter
        from matplotlib import pyplot as plt

        from nltk.tokenize import sent_tokenize
        from nltk.tokenize import word_tokenize

        import math
        import json
        import ast
        import random
        import os
        ```

        """
        # data columns
        self.datacols = ['unique_words', 'unique_lemmas', 'unique_bigrams', 'unique_lemma_bigrams', 'unique_bigrams_norel', 'unique_lemma_bigrams_norel']

        # load data if approved
        self._approve_data_dict(data)
        if not isinstance(model, stanza.Pipeline):
            print('WARN! this is a read-only version of Collocator')

        # model
        self.model = model
        # cleaner
        self.cleaner = cleaner

        # saving parameters
        self.interval_var = 0
        self.current_interval = 0
        self.language_code = language_code
        self.file_mode = file_mode if isinstance(file_mode, str) else 'csv'
        self.max_csv_len = max_csv_len if isinstance(max_csv_len, int) else 10000
        self.output_directory = output_directory if not output_directory or isinstance(output_directory, str) else None

        if self.output_directory:
            self._create_output_directory()

        # text reading parameters
        self.sentence_trim = sentence_trim if isinstance(sentence_trim, int) else None
        self.text_trim = text_trim if isinstance(text_trim, int) else None
        self.sentence_sampling = sentence_sampling if isinstance(sentence_sampling, str) else None

        # parsing parameters
        self.parse_deprel = parse_deprel if isinstance(parse_deprel, bool) else False

        # presentation parameters
        self.min_word_count = min_word_count if isinstance(min_word_count, int) else 3
        if self.min_word_count < 0:
            self.min_word_count = 3
        self.min_bigram_count = min_bigram_count if isinstance(min_bigram_count, int) else 3
        if self.min_bigram_count < 0:
            self.min_bigram_count = 3

        self.max_word_count = max_word_count if isinstance(max_word_count, int) else 30
        if self.max_word_count < 0:
            self.max_word_count = 30
        self.max_bigram_count = max_bigram_count if isinstance(max_bigram_count, int) else 30
        if self.max_bigram_count < 0:
            self.max_bigram_count = 30

    def _create_output_directory(self):
        try:
            if not os.path.isdir(self.output_directory):
                try:
                    os.mkdir(self.output_directory)
                except Exception as e:
                    print(e)
                    print('ERROR! something went wrong while creating an output directory')
                    raise Exception('Output directory name is invalid!')
        except Exception as e:
            print(e)
            raise Exception('Output directory name is invalid!')
        
    def xlx(self, f):
        return f * math.log(f) if f > 0 else 0

    def calculate_log_likelihood(self, bigram, bigram_counts, word_counts):
        """
        Рассчитывает log-likelihood для заданной биграммы.

        Args:
            bigram: кортеж (w1, w2) - биграмма для расчета
            bigram_counts: Counter биграмм в тексте
            word_counts: Counter уникальных слов в тексте
            total_words: Общее количество слов в корпусе

        Returns:
            Значение log-likelihood для данной биграммы
        """
        total_words = sum([value for value in word_counts.values()])
        w1, w2 = bigram

        # Частота биграммы (w1, w2)
        f_AB = bigram_counts.get(bigram, 0) # f_AB
        # Частота первого и второго слова
        f_A = word_counts.get(w1, 0)  # f_A
        f_B = word_counts.get(w2, 0)  # f_B
        N = total_words  # Общее число слов в корпусе

        # Проверка на нулевые значения
        if f_AB == 0 or f_A == 0 or f_B == 0 or N == 0:
            return float('-inf')

        # Вычисление log-likelihood по формуле
        log_likelihood = 2 * (
            self.xlx(f_AB) +
            self.xlx(f_A - f_AB) +
            self.xlx(f_B - f_AB) +
            self.xlx(N) +
            self.xlx(N + f_AB - f_A - f_B) -
            self.xlx(f_A) -
            self.xlx(f_B) -
            self.xlx(N - f_A) -
            self.xlx(N - f_B)
        )

        return log_likelihood

    def calculate_log_likelihood_lemma(self, bigram, bigram_counts, word_counts):
        """
        Рассчитывает log-likelihood для заданной биграммы.

        Args:
            bigram: кортеж (w1, w2) - биграмма для расчета
            bigram_counts: Counter биграмм в тексте
            word_counts: Counter уникальных слов в тексте
            total_words: Общее количество слов в корпусе

        Returns:
            Значение log-likelihood для данной биграммы
        """
        total_words = sum([value for value in word_counts.values()])
        w1, w2 = bigram

        # Частота биграммы (w1, w2)
        f_AB = bigram_counts.get(bigram, 0) # f_AB
        # Частота первого и второго слова
        f_A = word_counts.get(w1, 0)  # f_A
        f_B = word_counts.get(w2, 0)  # f_B
        N = total_words  # Общее число слов в корпусе

        # Проверка на нулевые значения
        if f_AB == 0 or f_A == 0 or f_B == 0 or N == 0:
            return float('-inf')

        # Вычисление log-likelihood по формуле
        log_likelihood = 2 * (
            self.xlx(f_AB) +
            self.xlx(f_A - f_AB) +
            self.xlx(f_B - f_AB) +
            self.xlx(N) +
            self.xlx(N + f_AB - f_A - f_B) -
            self.xlx(f_A) -
            self.xlx(f_B) -
            self.xlx(N - f_A) -
            self.xlx(N - f_B)
        )

        return log_likelihood

    def restrict_bigram_relation(self, relation):
        if not isinstance(relation, str):
            return

        # lists of bigrams to pop from dictionaries
        topop_bigrams = []
        topop_bigrams_norel = []
        topop_lemma_bigrams = []
        topop_lemma_bigrams_norel = []

        # fill lists
        for key, value in self.data['unique_bigrams'].items():
            bigram = (key[0], key[2])
            bigram_rel = key[1]
            if bigram_rel == relation:
                topop_bigrams.append(key)
                topop_bigrams_norel.append(bigram)
        for key, value in self.data['unique_lemma_bigrams'].items():
            bigram = (key[0], key[2])
            bigram_rel = key[1]
            if bigram_rel == relation:
                topop_lemma_bigrams.append(key)
                topop_lemma_bigrams_norel.append(bigram)

        # pop from dictionaries
        for bigram in topop_bigrams:
            self.data['unique_bigrams'].pop(bigram, None)
        for bigram in topop_lemma_bigrams:
            self.data['unique_lemma_bigrams'].pop(bigram, None)
        for bigram in topop_lemma_bigrams_norel:
            self.data['unique_lemma_bigrams_norel'].pop(bigram, None)
        for bigram in topop_bigrams_norel:
            self.data['unique_bigrams_norel'].pop(bigram, None)

        return

    def restrict_bigram_relation_vague(self, relation):
        if not isinstance(relation, str):
            return

        # lists of bigrams to pop from dictionaries
        topop_bigrams = []
        topop_bigrams_norel = []
        topop_lemma_bigrams = []
        topop_lemma_bigrams_norel = []

        # fill lists
        for key, value in self.data['unique_bigrams'].items():
            bigram = (key[0], key[2])
            bigram_rel = key[1]
            if relation in bigram_rel:
                topop_bigrams.append(key)
                topop_bigrams_norel.append(bigram)
        for key, value in self.data['unique_lemma_bigrams'].items():
            bigram = (key[0], key[2])
            bigram_rel = key[1]
            if relation in bigram_rel:
                topop_lemma_bigrams.append(key)
                topop_lemma_bigrams_norel.append(bigram)

        # pop from dictionaries
        for bigram in topop_bigrams:
            self.data['unique_bigrams'].pop(bigram, None)
        for bigram in topop_lemma_bigrams:
            self.data['unique_lemma_bigrams'].pop(bigram, None)
        for bigram in topop_lemma_bigrams_norel:
            self.data['unique_lemma_bigrams_norel'].pop(bigram, None)
        for bigram in topop_bigrams_norel:
            self.data['unique_bigrams_norel'].pop(bigram, None)

        return

    def filter_bigrams(self, minimum=2, maximum=20):
        self.data['unique_bigrams'] = {key: value for key, value in list(self.data['unique_bigrams'].items()) if self._validate_word(key) and self._validate_count(value, minimum=minimum) and self._validate_count_max(value, maximum=maximum)}
        self.data['unique_lemma_bigrams'] = {key: value for key, value in list(self.data['unique_lemma_bigrams'].items()) if self._validate_word(key) and self._validate_count(value, minimum=minimum) and self._validate_count_max(value, maximum=maximum)}

    def _validate_word(self, word):
        valid = []
        if not self.cleaner:
            return True
        if not isinstance(word, tuple):
            return False
        if len(word) < 3:
            return False
        if hasattr(self.cleaner, 'validate_word'):
            w1, w2 = word[0], word[2]
            for w in [w1, w2]:
                valid.append(self.cleaner.validate_word(w))
            return all(valid)
        return True

    def _validate_count(self, count, minimum=None):
        if not minimum:
            minimum = self.min_bigram_count
        if count <= minimum:
            return False
        return True

    def _validate_count_max(self, count, maximum=20):
        if count <= maximum:
            return True
        return False

    def _validate_couple(self, key, value):
        return all([self._validate_word(key), self._validate_count(value)])
    
    def get_dice_bigram(self, w1, w2):
        try:
            dice = (2 * self.get_bigram_count_norel(w1, w2)) / (self.get_word_count(w2) + self.get_word_count(w1))
            return dice
        except Exception as e:
            print(e, 'at bigram {}'.format((w1, w2)))
            return 0
        
    def get_dice_lemma_bigram(self, w1, w2):
        try:
            dice = (2 * self.get_bigram_count_lemma_norel(w1, w2)) / (self.get_lemma_count(w2) + self.get_lemma_count(w1))
            return dice
        except Exception as e:
            print(e, 'at bigram {}'.format((w1, w2)))
            return 0

    def present_statistics(self):
        collocation_tuples = [(key, value) for key, value in self.data['unique_bigrams'].items() if self._validate_couple(key, value)]
        collocations = [key for key, value in collocation_tuples]
        print('collocations array len: ', len(collocations))
        collocation_counts = [self.get_bigram_count_norel(key[0], key[2]) for key, value in collocation_tuples]
        print('collocation counts array len: ', len(collocation_counts))
        word1_counts = [self.get_word_count(collo[0]) for collo in collocations]
        word2_counts = [self.get_word_count(collo[2]) for collo in collocations]

        # Statistics
        collocation_dice = [self.get_dice_bigram(collo[0], collo[2]) for collo in collocations]
        collocation_loglikelihood = [self.get_log_likelihood(collo[0], collo[2]) for collo in collocations]

        # DataFrame
        our_data = {
            'collocation': collocations,
            'rel type': [collocation[1] for collocation in collocations]
        }
        if self.parse_deprel:
            try:
                our_data['deprel type'] = [collocation[3] for collocation in collocations]
            except Exception as e:
                print(e)
                print('parse_deprel mode is active, but no dep relations are found')
        our_data['count'] = collocation_counts
        our_data['word 1 count'] = word1_counts
        our_data['word 2 count'] = word2_counts
        our_data['dice'] = collocation_dice
        our_data['log-likelihood'] = collocation_loglikelihood
        
        return pd.DataFrame(our_data)
    
    def present_statistics_unsafe(self):
        collocation_tuples = [(key, value) for key, value in self.data['unique_bigrams'].items()]
        collocations = [key for key, value in collocation_tuples]
        print('collocations array len: ', len(collocations))
        collocation_counts = [self.get_bigram_count_norel(key[0], key[2]) for key, value in collocation_tuples]
        print('collocation counts array len: ', len(collocation_counts))
        word1_counts = [self.get_word_count(collo[0]) for collo in collocations]
        word2_counts = [self.get_word_count(collo[2]) for collo in collocations]

        # Statistics
        collocation_dice = [self.get_dice_bigram(collo[0], collo[2]) for collo in collocations]
        collocation_loglikelihood = [self.get_log_likelihood(collo[0], collo[2]) for collo in collocations]

        # DataFrame
        our_data = {
            'collocation': collocations,
            'rel type': [collocation[1] for collocation in collocations]
        }
        if self.parse_deprel:
            try:
                our_data['deprel type'] = [collocation[3] for collocation in collocations]
            except Exception as e:
                print(e)
                print('parse_deprel mode is active, but no dep relations are found')
        our_data['count'] = collocation_counts
        our_data['word 1 count'] = word1_counts
        our_data['word 2 count'] = word2_counts
        our_data['dice'] = collocation_dice
        our_data['log-likelihood'] = collocation_loglikelihood
        
        return pd.DataFrame(our_data)

    def present_statistics_by_relation(self, relation):
        df = self.present_statistics()
        return df[df['rel type'] == relation]
    
    def save_statistics_unsafe(self):
        df = self.present_statistics_unsafe()
        self._save_csv_formatted(df, 'saved_statistics_{}.csv')
    
    def _prepare_output_folder(self):
        if not os.path.isdir('output'):
            os.mkdir('output')
    
    def _save_csv_formatted(self, data, name):
        self._prepare_output_folder()
        name = name.format(self.language_code)
        if self.output_directory:
            stats_dir = self.output_directory + '/' + 'statistics'
            if not os.path.isdir(stats_dir):
                os.mkdir(stats_dir)
            name = stats_dir + '/' + name
        data.to_csv(name)
        print('Statistics saved at {}'.format(name))

    def save_statistics(self):
        df = self.present_statistics()
        self._save_csv_formatted(df, 'saved_statistics_{}.csv')

    def present_statistics_lemma(self):
        collocation_tuples = [(key, value) for key, value in self.data['unique_lemma_bigrams'].items() if self._validate_couple(key, value)]
        collocations = [key for key, value in collocation_tuples]
        print('collocations array len: ', len(collocations))
        collocation_counts = [self.get_bigram_count_lemma_norel(key[0], key[2]) for key, value in collocation_tuples]
        print('collocation counts array len: ', len(collocation_counts))
        word1_counts = [self.get_lemma_count(collo[0]) for collo in collocations]
        word2_counts = [self.get_lemma_count(collo[2]) for collo in collocations]

        # Statistics
        collocation_dice = [self.get_dice_lemma_bigram(collo[0], collo[2]) for collo in collocations]
        collocation_loglikelihood = [self.get_log_likelihood_lemma(collo[0], collo[2]) for collo in collocations]

        our_data = {
            'collocation': collocations,
            'rel type': [collocation[1] for collocation in collocations]
        }
        if self.parse_deprel:
            try:
                our_data['deprel type'] = [collocation[3] for collocation in collocations]
            except Exception as e:
                print(e)
                print('parse_deprel mode is active, but no dep relations are found')
        our_data['count'] = collocation_counts
        our_data['word 1 count'] = word1_counts
        our_data['word 2 count'] = word2_counts
        our_data['dice'] = collocation_dice
        our_data['log-likelihood'] = collocation_loglikelihood

        return pd.DataFrame(our_data)

    def present_statistics_lemma_unsafe(self):
        collocation_tuples = [(key, value) for key, value in self.data['unique_lemma_bigrams'].items()]
        collocations = [key for key, value in collocation_tuples]
        print('collocations array len: ', len(collocations))
        collocation_counts = [self.get_bigram_count_lemma_norel(key[0], key[2]) for key, value in collocation_tuples]
        print('collocation counts array len: ', len(collocation_counts))
        word1_counts = [self.get_lemma_count(collo[0]) for collo in collocations]
        word2_counts = [self.get_lemma_count(collo[2]) for collo in collocations]

        # Statistics
        collocation_dice = [self.get_dice_lemma_bigram(collo[0], collo[2]) for collo in collocations]
        collocation_loglikelihood = [self.get_log_likelihood_lemma(collo[0], collo[2]) for collo in collocations]

        our_data = {
            'collocation': collocations,
            'rel type': [collocation[1] for collocation in collocations]
        }
        if self.parse_deprel:
            try:
                our_data['deprel type'] = [collocation[3] for collocation in collocations]
            except Exception as e:
                print(e)
                print('parse_deprel mode is active, but no dep relations are found')
        our_data['count'] = collocation_counts
        our_data['word 1 count'] = word1_counts
        our_data['word 2 count'] = word2_counts
        our_data['dice'] = collocation_dice
        our_data['log-likelihood'] = collocation_loglikelihood

        return pd.DataFrame(our_data)
        
    def save_statistics_lemma(self):
        df = self.present_statistics_lemma()
        self._save_csv_formatted(df, 'saved_statistics_lemma_{}.csv')

    def present_count_histogram(self):
        numbers = list(self.data['unique_bigrams'].values())
        plt.hist(numbers,
                 bins=range(min(numbers), max(numbers) + 2),
                 label='Распределение частот биграмм по пройденным текстам'),
        plt.show()

    def _approve_data_dict(self, data):
        if isinstance(data, dict):
            self.data = data
            if not self._validate_data(self.data):
                self._create_data_dict()
        else:
            self._create_data_dict()

    def _create_data_dict(self):
        self.data = dict()
        for counter in self.datacols:
            data_part = self.data.get(counter, None)
            if not data_part or not isinstance(data_part, Counter):
                self.data[counter] = Counter()

    def get_word_count(self, word):
        return self.data['unique_words'].get(word, 0)

    def get_lemma_count(self, word):
        return self.data['unique_lemmas'].get(word, 0)

    def get_bigram_count(self, w1, r, w2):
        return self.data['unique_bigrams'].get((w1, r, w2), 0)

    def get_bigram_count_norel(self, w1, w2):
        return self.data['unique_bigrams_norel'].get((w1, w2), 0)

    def get_bigram_count_lemma(self, l1, r, l2):
        return self.data['unique_lemma_bigrams'].get((l1, r, l2), 0)

    def get_bigram_count_lemma_norel(self, l1, l2):
        return self.data['unique_lemma_bigrams_norel'].get((l1, l2), 0)

    def get_similar_ngrams(self, w1, w2):
        return [ngram for ngram, count in self.data['unique_bigrams'].items() if ngram[0] == w1 and ngram[2] == w2]

    def get_similar_ngrams_lemma(self, l1, l2):
        return [ngram for ngram, count in self.data['unique_lemma_bigrams'].items() if ngram[0] == l1 and ngram[2] == l2]

    def get_w1_r_count(self, w1, r):
        return sum([count for ngram, count in self.data['unique_bigrams'].items() if ngram[0] == w1 and ngram[1] == r])

    def get_w1_r_count_lemma(self, w1, r):
        return sum([count for ngram, count in self.data['unique_lemma_bigrams'].items() if ngram[0] == w1 and ngram[1] == r])

    def get_anyr_w2_count(self, w2):
        return sum([count for ngram, count in self.data['unique_bigrams'].items() if ngram[2] == w2])

    def get_anyr_w2_count_lemma(self, w2):
        return sum([count for ngram, count in self.data['unique_lemma_bigrams'].items() if ngram[2] == w2])

    def get_log_dice_bigram(self, w1, r, w2):
        logdice = 0
        try:
            logdice = 14 + math.log2((2*self.get_bigram_count(w1, r, w2))/(self.get_w1_r_count(w1, r) + self.get_anyr_w2_count(w2)))
        except ValueError as e:
            logdice = 0
        return logdice if logdice > 0 else float('inf')

    def get_log_dice_bigram_lemma(self, w1, r, w2):
        logdice = 0
        try:
            logdice = 14 + math.log2((2*self.get_bigram_count_lemma(w1, r, w2))/(self.get_w1_r_count_lemma(w1, r) + self.get_anyr_w2_count_lemma(w2)))
        except Exception as e:
            logdice = 0
        return logdice if logdice > 0 else float('inf')

    def get_log_likelihood(self, w1, w2):
        # (bigram, bigram_counts, word_counts, total_words)
        try:
            log_likelihood = self.calculate_log_likelihood((w1, w2), self.data['unique_bigrams_norel'], self.data['unique_words'])
            return log_likelihood
        except Exception as e:
            print(e)
            return 0

    def get_log_likelihood_lemma(self, w1, w2):
        try:
            log_likelihood = self.calculate_log_likelihood((w1, w2), self.data['unique_lemma_bigrams_norel'], self.data['unique_lemmas'])
            return log_likelihood
        except Exception as e:
            print(e)
            return 0

    def _calcprob(self, bigram_count, unigram_count):
        if unigram_count > 0:
            if bigram_count in [0, 0.0]:
                print('bigram count equals to zero')
            return bigram_count / unigram_count
        else:
            print('unigram not found')
        return 0

    def _getw(self, token):
        return token.get('word', None)

    def _getl(self, token):
        return token.get('lemma', None)

    def _getp(self, token):
        return token.get('pos', None)

    def _getdep(self, token):
        return token.get('deprel', None)

    def _addword(self, word):
        self.data['unique_words'].update([word])

    def _addlemma(self, lemma):
        self.data['unique_lemmas'].update([lemma])

    def _addbigram(self, bigram):
        self.data['unique_bigrams'].update([bigram])

    def _addbigramnorel(self, bigram):
        self.data['unique_bigrams_norel'].update([bigram])

    def _addlemmabigram(self, lemmabigram):
        self.data['unique_lemma_bigrams'].update([lemmabigram])

    def _addlemmabigramnorel(self, lemmabigram):
        self.data['unique_lemma_bigrams_norel'].update([lemmabigram])

    def _addrelation(self, relation):
        self.data['unique_relations'].update([relation])

    def _gettoken(self, token):
        """returns dictionary with keys 'word', 'lemma' and 'pos' for token"""
        if not self.parse_deprel:
            return {'word': token.text.lower(),
                    'lemma': token.lemma.lower(),
                    'pos': token.upos.lower()}
        else:
            return {'word': token.text.lower(),
                    'lemma': token.lemma.lower(),
                    'pos': token.upos.lower(),
                    'deprel': token.deprel.lower()}

    def _gettokens(self, sentence):
        """returns a list of dictionaries with keys 'word', 'lemma' and 'pos' for tokens"""
        return [self._gettoken(token) for token in sentence.words if token.upos != 'PUNCT']

    def _getdoc(self, text):
        return self.model(text)

    def _getsents(self, text):
        return self._getdoc(text).sentences

    def _ngram(self, tokens, n=2):
        ngrams = []
        for i in range(len(tokens)-n+1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    def _ngram_relation(self, ngram):
        upos1 = self._getp(ngram[0])
        upos2 = self._getp(ngram[1])
        return '{}-{}'.format(upos1, upos2)

    def _ngram_deprel(self, ngram):
        deprel1 = self._getdep(ngram[0])
        deprel2 = self._getdep(ngram[1])
        return '{}-{}'.format(deprel1, deprel2)

    def _get_bigramw(self, bigram):
        return self._getw(bigram[0]), self._getw(bigram[1])

    def _get_bigraml(self, bigram):
        return self._getl(bigram[0]), self._getl(bigram[1])

    def _add_bigram_r(self, ngram, r):
        word1, word2 = self._get_bigramw(ngram)
        if not self.parse_deprel:
            assert isinstance(r, str), "relation datatype should be string"
            self._addbigram((word1, r, word2))
        else:
            assert isinstance(r, list), "relation datatype should be list"
            assert len(r) == 2, "relation list len is not correct. should be 2"
            assert all([isinstance(s, str) for s in r]), "all relation instances must be strings"
            self._addbigram((word1, r[0], word2, r[1]))

    def _add_lemmabigram_r(self, ngram, r):
        lemma1, lemma2 = self._get_bigraml(ngram)
        if not self.parse_deprel:
            assert isinstance(r, str), "relation datatype should be string"
            self._addlemmabigram((lemma1, r, lemma2))
        else:
            assert isinstance(r, list), "relation datatype should be list"
            assert len(r) == 2, "relation list len is not correct. should be 2"
            assert all([isinstance(s, str) for s in r]), "all relation instances must be strings"
            self._addlemmabigram((lemma1, r[0], lemma2, r[1]))

    def _add_bigram_norel(self, ngram):
        word1, word2 = self._get_bigramw(ngram)
        self._addbigramnorel((word1, word2))

    def _add_lemma_bigram_norel(self, ngram):
        lemma1, lemma2 = self._get_bigraml(ngram)
        self._addlemmabigramnorel((lemma1, lemma2))

    def _process_ngrams(self, tokens):
        for ngram in self._ngram(tokens):
            # ({}, {})
            self._add_bigram_norel(ngram)
            self._add_lemma_bigram_norel(ngram)
            if not self.parse_deprel:
                r = self._ngram_relation(ngram)
            else:
                r = [self._ngram_relation(ngram), self._ngram_deprel(ngram)]
            self._add_bigram_r(ngram, r)
            self._add_lemmabigram_r(ngram, r)

    def _validate_data(self, data):
        if len(list(data.keys())) < len(self.datacols):
            return False
        if any([n not in data.keys() for n in self.datacols]):
            return False
        return True
    
    def _load_csv_key(self, datakey, csv_files):
        data_counter = Counter()
        datakey_dunder = "__" + datakey + "__"
        for file in csv_files:
            if datakey_dunder in file:
                csv_data = pd.read_csv(file)
                for index, row in csv_data.iterrows():
                    if datakey in self.datacols[:2]:
                        data_counter[row['key']] = row['count']
                    if datakey in self.datacols[2:]:
                        data_counter[ast.literal_eval(row['key'])] = row['count']
        return data_counter
    
    def _load_csv(self, foldername):
        if not os.path.isdir(foldername):
            raise Exception('Choose an existing folder')
        # Готовим список для файлов .csv
        found_csv_files = []
        # Готовим словарь для новых данных
        new_data = {key: Counter() for key in self.datacols}
        # Находим все файлы .csv
        for root, dirs, files in os.walk(foldername):
            for file in files:
                if '.csv' in file:
                    found_csv_files.append(os.path.join(root, file))
        # Проходимся по всем столбцам данных
        for key in self.datacols:
            # Обновляем данные по этой "колонке" из подходящих файлов .csv, которые нам встретились
            new_data[key].update(self._load_csv_key(key, found_csv_files))
        # Проверяем, что данные подходят для использования
        if self._validate_data(new_data):
            self.data = new_data
        else:
            self.data = dict()
        print('data load complete')
        print('unique words : {}'.format(len(list(self.data['unique_words'].items()))))
        print('unique bigrams : {}'.format(len(list(self.data['unique_bigrams'].items()))))
        return True

    def _load_json(self, filename):
        with open(f'{filename}', mode='r', encoding='utf-8') as f:
            load_data = json.load(f)
        print('')
        if not self._validate_data(load_data):
            print('WARN! data validation fail on load')
            self.data = dict()
        else:
            for col in self.datacols[:2]:
                self.data[col] = load_data[col]
            for col in self.datacols[2:]:
                self.data[col] = {ast.literal_eval(key): value for key, value in load_data[col].items()}

    def _load_pickle(self, filename):
        """TODO NOTE read documentation"""
        with open(filename, 'rb') as f:
            new_data = pickle.load(f)
            if self._approve_data_dict(new_data):
                self.data = new_data 

    def load(self, filename):
        if self.file_mode == 'json':
            self._load_json(filename)
            return
        elif self.file_mode == 'csv':
            self._load_csv(filename)
            return
        self._load_pickle(filename)

    def _format_file_name_for_saving(self, filename):
        return f'{filename}_{self.language_code}'

    def _save_json(self, filename):
        """NOTE Deprecated"""
        # Putting a file inside an output_directory if one is available
        if self.output_directory:
            filename = self.output_directory + '/' + filename
        # Saving JSON FILE
        with open(f'{self._format_file_name_for_saving(filename)}.json', mode='w', encoding='utf-8') as f:
            saved_data = {key: {str(k): v for k, v in value.items()} for key, value in self.data.items()}
            json.dump(saved_data, f, indent='\t', ensure_ascii=False)
            print('Save complete at {}'.format(filename))

    def _check_save_folder_for_csv(self, foldername):
        if not os.path.exists(foldername):
            os.mkdir(foldername)

    def _dump_csv(self, file_code, saving_dict, current_num):
        pd.DataFrame(saving_dict).to_csv(file_code + f"_{current_num}.csv")

    def _save_data_key_csv(self, datakey, datadict, folder):
        csv_file_code = folder + "/" + "__" +  datakey + "_"
        saving_dict = {
            'key': [],
            'count': []
        }
        saved_values = 0
        current_csv_file = 0
        for key, count in datadict.items():
            saving_dict['key'].append(str(key))
            saving_dict['count'].append(count)
            saved_values += 1
            if saved_values >= self.max_csv_len:
                self._dump_csv(csv_file_code, saving_dict, current_csv_file)
                saved_values = 0
                current_csv_file += 1
                saving_dict = {
                                'key': [],
                                'count': []
                               }
        if saved_values > 0:
            self._dump_csv(csv_file_code, saving_dict, current_csv_file)
        return True

    def _save_csv(self, foldername):
        # Making a folder inside an output_folder if self.output_directory is stated
        if self.output_directory:
            foldername = self.output_directory + '/' + foldername
        self._check_save_folder_for_csv(foldername)
        # Saving CSV Data in that folder
        for key, value in self.data.items():
            self._save_data_key_csv(key, value, foldername)

    def _save_pickle(self, filename):
        """TODO NOTE read pickle documentation"""
        if self.output_directory:
            filename = self.output_directory + '/' + filename
        # Preparing a .pkl file for our needs
        with open(f'{self._format_file_name_for_saving(filename)}.pkl', mode='wb') as f:
            pickle.dump(self.data, f)  

    def save(self, filename):
        if self.file_mode == 'json':
            self._save_json(filename)
            return
        elif self.file_mode == 'csv':
            self._save_csv(filename)
            return
        self._save_pickle(filename)

    def _feed(self, text):
        for sentence in self._getsents(text):
            # dictionaries with keys 'word', 'lemma', 'pos' for each word
            sentence = self._gettokens(sentence)
            for token in sentence:
                self._addword(self._getw(token))
                self._addlemma(self._getl(token))
            self._process_ngrams(sentence)

    def feed(self, text):
        sentences = sent_tokenize(text)
        if self.text_trim:
            if not self.sentence_sampling or self.sentence_sampling == 'start':
                sentences = sentences[:self.text_trim]
            if self.sentence_sampling == 'random':
                new_sentences = []
                for i in range(self.text_trim):
                    try:
                        sentence = random.choice(sentences)
                        sentences.remove(sentence)
                        new_sentences.append(sentence)
                    except Exception as e:
                        break
                sentences = new_sentences
                new_sentences = None
        for sentence in sentences:
            if self.cleaner:
                if hasattr(self.cleaner, 'clean_text'):
                    sentence = self.cleaner.clean_text(sentence)
            if self.sentence_trim:
                sentence = ' '.join(word_tokenize(sentence)[:self.sentence_trim])
            self._feed(sentence)

    def __call__(self, text, save_interval=100):
        # Read only mode
        if not self.model:
            print('WARN! Collocator in read-only mode, no stanza model initialized')
            return
        # Text validation
        if isinstance(text, str):
            self.feed(text)
        else:
            raise ValueError('wrong type of text')
        # Intervals and save
        self.interval_var += 1
        if self.interval_var >= save_interval:
            self.save('interval_save_{}'.format(self.current_interval))
            self.interval_var = 0
            self.current_interval += 1

import re
from string import punctuation

# CLEANER
class Cleaner:
    def __init__(self,
                 stopwords=None,
                 regex=None,
                 filter_numbers=False,
                 filter_punctuation=False,
                 min_word_len=None):

        # stopwords
        self.stopwords = set()
        for stopword in stopwords:
            if ' ' in stopword:
                self.stopwords.update(stopword.split(' '))
            else:
                self.stopwords.add(stopword)
        # regex
        self.set_regex(regex)

        # additional word filtering
        self.filter_numbers = filter_numbers if isinstance(filter_numbers, bool) else False
        self.filter_punctuation = filter_punctuation if isinstance(filter_punctuation, bool) else False
        self.min_word_len = min_word_len if isinstance(min_word_len, int) else None

        # default package
        self.numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # debugging
        self.debug_log = []

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords if isinstance(stopwords, list) else None
        if isinstance(self.stopwords, list):
            self.stopwords = [n for n in self.stopwords if isinstance(n, str)]

    def set_regex(self, regex):
        self.regex = regex if isinstance(regex, list) else None
        if isinstance(self.regex, list):
            self.regex = [r for r in self.regex if isinstance(r, re.Pattern)]

    def replace_double_spaces(self, input_string):
        return re.sub(r'\s{2,}', ' ', input_string)

    def clean_text(self, text):
        if self.regex:
            for reg in self.regex:
                text = reg.sub('', text)
        return text

    def validate_word(self, word):
        for stopword in self.stopwords:
            if stopword == word:
                return False
            if self.filter_numbers and any([number in word for number in self.numbers]):
                return False
            if self.min_word_len:
                if len(word) < self.min_word_len:
                    return False
            if self.filter_punctuation:
                for symbol in punctuation:
                    if symbol in word:
                        self.debug_log.append(f'<!> rejected {word} :: symbol -- {symbol} -- present in word <!>')
                        return False
        return True

    def _process_dict_with_stopwords(self, new_dict, old_dict):
        for key, value in old_dict.items():
            valid = True
            for stopword in self.stopwords:
                if stopword in key:
                    valid = False
            if valid:
                new_dict[key] = value
        return new_dict

    def clean_dict(self, dictionary):
        new_dict = dict()
        if self.stopwords:
            dictionary = self._process_dict_with_stopwords(new_dict, dictionary)
        return dictionary

import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords

LANGUAGES = ['ru', 'kk']

STOPWORDS = {
    'ru': stopwords.words('russian'),
    'kk': stopwords.words('kazakh')
}

russian_stopwords = [
    "с", "когда", "она", "он", "они", "мы", "я", "ты", "уже", "и", "хотя", "из", "но", "это", "в", "по", "на", "к", "о", "об", "за", "для",
    "а", "ж", "же", "все", "всё", "всего", "его", "ему", "ее", "её", "ей", "ею", "их", "им", "ими", "оно", "тем", "тому", "эти", "этого", "этом", "этим", "этой",
    "где", "куда", "откуда", "тогда", "затем", "потом", "после", "перед", "до", "во", "не", "ни", "как", "что", "чем", "будет", "бы", "был", "была", "были", "было",
    "чтобы", "то", "ту", "те", "тут", "здесь", "там", "туда", "оттуда", "сюда", "себя", "весь", "всех", "всей", "всем", "всеми", "всему", "какой", "какая", "какие", "каким",
    "какому", "какою", "чей", "чья", "чьи", "чьим", "чьему", "чью", "сам", "сама", "сами", "само", "самым", "самой", "самим",
    "ли", "нибудь", "либо", "даже", "только", "хоть", "ну", "да", "нет", "может", "можно", "нужно", "надо", "следует", "должен", "должна", "должны", "должно",
    "этот", "эта", "еще", "ещё", "раз", "опять", "снова", "уж", "более", "менее", "очень", "слишком", "достаточно", "много", "мало", "немного", "сколько", "примерно",
    "почти", "совсем", "абсолютно", "обычно", "часто", "иногда", "редко", "всегда", "никогда", "просто", "легко", "трудно", "сложно", "важно", "необходимо",
    "обязательно", "возможно", "вероятно", "скорее", "целом", "итак", "правило", "крайней", "мере", "числе", "связи", "поскольку", "так", "если", "иначе",
    "говоря", "иными", "словами", "есть", "вместо", "того", "чтобы", "благодаря", "вопреки", "несмотря", "счет", "именно", "лишь", "исключительно",
    "тд", "тп", "например", "также", "кроме", "помимо", "этого", "наряду", "этим", "однако", "между", "временем", "зависимости", "или", "иного",
    "силу", "причине", "этой", "из-за", "ввиду", "том", "касается", "чего", "отношении", "поводу", "рамках", "сфере", "части", "нас", "во",
    "всяком", "случае", "противном", "при", "менее",  "которые", "которая", "которое", "этот", "эти", "отнюдь", "какие то", "какаято",
    "какойто", "вдоль", "любой", "любая", "любое", "кому", "вашу", "ваши", "ваша", "ваш", "всё", "какими", "какого", "каким", "ко", "свой", "своя", "свое", "около",
    "таков", "такая", "такое", "столь", "другой", "другая", "другое", "некий", "некая", "некое", "нечто", "всякий", "всякая", "всякое", "каждый", "каждая", "каждое", "наверное", "вероятно", "поэтому", "впрочем", "вследствие"
]

STOPWORDS['ru'].extend(russian_stopwords)

custom_stop_words = {
    'мен', 'сен', 'сіз', 'ол', 'біз', 'сендер', 'сіздер', 'олар',
    'бұл', 'осы', 'мына', 'мынау', 'сол', 'ол', 'ана', 'анау',
    'міне', 'әні', 'әне', 'әнеки', 'сонау', 'және', 'да', 'де',
    'та', 'те', 'әрі', 'сондай-ақ', 'тағы', 'әлде', 'бірақ',
    'алайда', 'ал', 'дегенмен', 'әйтпесе', 'немесе', 'бірде',
    'кейде', 'себебі', 'өйткені', 'сондықтан', 'себепті',
    'үшін', 'деп', 'егер', 'онда', 'егерде', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', 'нөл', 'бір', 'екі', 'үш', 'төрт', 'бес',
    'алты', 'жеті', 'сегіз', 'тоғыз', 'он', 'он бір', 'он екі',
    'он үш', 'он төрт', 'он бес', 'он алты', 'он жеті', 'он сегіз',
    'он тоғыз', 'жиырма', 'жиырма бір', 'жиырма екі', 'жиырма үш',
    'жиырма төрт', 'жиырма бес', 'жиырма алты', 'жиырма жеті',
    'жиырма сегіз', 'жиырма тоғыз', 'отыз', 'отыз бір', 'отыз екі',
    'отыз үш', 'отыз төрт', 'отыз бес', 'отыз алты', 'отыз жеті',
    'отыз сегіз', 'отыз тоғыз', 'қырық', 'қырық бір', 'қырық екі',
    'қырық үш', 'қырық төрт', 'қырық бес', 'қырық алты', 'қырық жеті',
    'қырық сегіз', 'қырық тоғыз', 'елу', 'елу бір', 'елу екі',
    'елу үш', 'елу төрт', 'елу бес', 'елу алты', 'елу жеті',
    'елу сегіз', 'елу тоғыз', 'алпыс', 'алпыс бір', 'алпыс екі',
    'алпыс үш', 'алпыс төрт', 'алпыс бес', 'алпыс алты', 'алпыс жеті',
    'алпыс сегіз', 'алпыс тоғыз', 'жетпіс', 'жетпіс бір', 'жетпіс екі',
    'жетпіс үш', 'жетпіс төрт', 'жетпіс бес', 'жетпіс алты',
    'жетпіс жеті', 'жетпіс сегіз', 'жетпіс тоғыз', 'сексен',
    'сексен бір', 'сексен екі', 'сексен үш', 'сексен төрт',
    'сексен бес', 'сексен алты', 'сексен жеті', 'сексен сегіз',
    'сексен тоғыз', 'тоқсан', 'тоқсан бір', 'тоқсан екі',
    'тоқсан үш', 'тоқсан төрт', 'тоқсан бес', 'тоқсан алты',
    'тоқсан жеті', 'тоқсан сегіз', 'тоқсан тоғыз', 'жүз', 'мың',
    'сонымен', 'қатар', 'сияқты', 'туралы', 'жайлы', 'арқылы', 'сайын',
    'кейін', 'бұрын', 'бойынша', 'қарсы', 'қарай', 'себепті', 'арқасында',
    'бойы', 'ішінде', 'сыртында', 'алдында', 'жанында', 'арасында',
    'қасында', 'үстінде', 'астында', 'жолында', 'мүмкін', 'әрине',
    'шынында', 'демек', 'мысалы', 'әдетте', 'әсіресе', 'біріншіден',
    'екіншіден', 'қорытындылай', 'келгенде', 'қысқасы', 'өте', 'тым',
    'тіпті', 'әлі', 'енді', 'қайтадан', 'мүлдем', 'бірден', 'бірге',
    'біршама', 'әдетте', 'нәрсе', 'жағдай', 'мәселе', 'туралы', 'жайында',
    'сияқты'

}

STOPWORDS['kk'].extend(custom_stop_words)

# REGEX (WORK IN PROGRESS)

regex_camel_case = re.compile(r'\b[а-яё]+[А-ЯЁ]+[а-яёА-ЯЁ]+\b')
regex_1st_letter_capital = re.compile(r'\b[А-ЯЁ]+[а-яё]+\b')
regex_all_caps = re.compile(r'\b[А-ЯЁ]+\b')
regex_hashtag = re.compile(r'#[a-zA-Zа-яА-ЯёЁ\-\_\d]+')
regex_url = re.compile(r'(https?://(?:www.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9].[^\s]{2,}|www.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9].[^\s]{2,}|https?://(?:www.|(?!www))[a-zA-Z0-9]+.[^\s]{2,}|www.[a-zA-Z0-9]+.[^\s]{2,})')
regex_date = re.compile(r'\b\d{2}[-/\.]\d{2}-/\.\b')
regex_time = re.compile(r'\b\d{1,2}:\d{2}\b')
regex_email = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
regex_eng = re.compile(r'\b[a-zA-Z]+\b')
regex_ampersand = re.compile(r'&[a-zA-Z0-9;#]+;')
regex_concecutive_nonalphanum = re.compile(r'[^a-zA-Z\d\sа-яА-ЯёЁ#*]{2,}')
regex_date = re.compile(r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b')
regex_special_chars = re.compile(r'[|\\/*]')
regex_kz_months = re.compile(r'\b(?:Қаңтар|Ақпан|Наурыз|Сәуір|Май|Маусым|Шілде|Тамыз|Қыркүйек|Қазан|Қараша|Желтоқсан|қаңтар|ақпан|наурыз|сәуір|май|маусым|шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)\b')
regex_kz_weekdays = re.compile(r'\b(?:Дүйсенбi|Сейсенбi|Сәрсенбi|Бейсенбi|Жұма|Сенбi|Жексенбi|дүйсенбi|сейсенбi|сәрсенбi|бейсенбi|жұма|сенбi|жексенбi)\b')
regex_kz_domains = re.compile(r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.(?:kz)\b', re.IGNORECASE)
regex_month_day_year_time = re.compile(r'\b(?:Қаңтар|Ақпан|Наурыз|Сәуір|Май|Маусым|Шілде|Тамыз|Қыркүйек|Қазан|Қараша|Желтоқсан) \d{1,2}, \d{4},? \d{1,2}:\d{2}\b')
regex_year_month_day_pipe = re.compile(r'\b\d{4},? (?:Қаңтар|Ақпан|Наурыз|Сәуір|Май|Маусым|Шілде|Тамыз|Қыркүйек|Қазан|Қараша|Желтоқсан) \d{1,2} \|\b')
regex_info_domains = re.compile(r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.(?:info)\b')
regex_main_news = re.compile(r'\bГлавная Новости\b')
regex_day_month_time = re.compile(r'\b\d{1,2} (?:Қаңтар|Ақпан|Наурыз|Сәуір|Май|Маусым|Шілде|Тамыз|Қыркүйек|Қазан|Қараша|Желтоқсан) \d{1,2}:\d{2}\b')
regex_weekday_day_month_year_time = re.compile(r'\b(?:Дүйсенбi|Сейсенбi|Сәрсенбi|Бейсенбi|Жұма|Сенбi|Жексенбi), \d{1,2} (?:Қаңтар|Ақпан|Наурыз|Сәуір|Май|Маусым|Шілде|Тамыз|Қыркүйек|Қазан|Қараша|Желтоқсан), \d{4}(?: \d{1,2}:\d{2})?\b')
regex_datetime = re.compile(r'\b\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\b')
regex_kz_sites = re.compile(r'\b\w+kaz\.\w+\.kz\b', re.UNICODE)
regex_quote = re.compile(r'»')
regex_dash = re.compile(r'–')
regex_list_letters = re.compile(r'\b[а-әa-zA-Z]\)\b')
regex_decimal_numbers = re.compile(r'\b\d+\.\d+\b')
regex_news_header = re.compile(r'\bБасты бет Жаңалықтар\b', re.UNICODE)
regex_time_kz = re.compile(r'\b\d{1,2}:\d{2}-(де|да|те|та)\b', re.UNICODE)
regex_signed_numbers = re.compile(r'\b[+-]\d+\b')
regex_number_kz = re.compile(r'\b\d{1,}-(де|да|те|та|ден|дан|тен|тан)\b', re.UNICODE)
regex_latin = re.compile(r'\b[a-zA-Z]+\b', re.UNICODE)
regex_numbers = re.compile(r'\b\d+\b')
regex_men = re.compile(r'\bмен\b', re.UNICODE)  # Убираем "мен"
regex_spaces = re.compile(r'\s+', re.UNICODE)  # Убираем лишние пробелы
regex_punctuation = re.compile(r'[^\w\s]', re.UNICODE)  # Убираем знаки препинания
regex_digits = re.compile(r'\d+', re.UNICODE)  # Убираем числа
regex_latin_en = re.compile(r'\b([a-zA-Z]|[0-9])+\b', re.UNICODE)

REGEX = {lang: [regex_camel_case, regex_hashtag, regex_email, regex_url, regex_date, regex_time, regex_punctuation] for lang in LANGUAGES}

REGEX['ru'].extend([regex_eng, regex_numbers, regex_latin_en, regex_digits])

REGEX['kk'] = [
    regex_hashtag,
    regex_url,
    regex_email,
    regex_time,
    regex_eng,
    regex_special_chars,
    regex_kz_months,
    regex_kz_weekdays,
    regex_kz_domains,
    regex_month_day_year_time,
    regex_year_month_day_pipe,
    regex_info_domains,
    regex_main_news,
    regex_day_month_time,
    regex_weekday_day_month_year_time,
    regex_datetime,
    regex_kz_sites,
    regex_quote,
    regex_dash,
    regex_list_letters,
    regex_decimal_numbers,
    regex_news_header,
    regex_time_kz,
    regex_signed_numbers,
    regex_number_kz,
    regex_latin,
    regex_men

]

class stopx:
    def words(self, language):
        return STOPWORDS.get(language, list())

class regx:
    def regex(self, language):
        return REGEX.get(language, list())
    
# WORK SECTION
LANGUAGE = 'en'

stanza.download(LANGUAGE)
FILENAME = "output_data_en_1000.csv"
DATA = pd.read_csv(FILENAME)
MAX_DATA = 1000 * 100

coll = CollocatorDeluxe(stanza.Pipeline(LANGUAGE, use_gpu=False),
                        cleaner=Cleaner(stopwords=[],
                                        regex=regx().regex(LANGUAGE),
                                        filter_numbers=True,
                                        filter_punctuation=True,
                                        min_word_len = 2),
                        language_code=LANGUAGE,
                        parse_deprel=True,
                        text_trim=5,
                        sentence_sampling='random',
                        sentence_trim=20,
                        min_bigram_count=3,
                        file_mode='pickle',
                        output_directory='stats_{}'.format(LANGUAGE))

def process_data(args):
    """Функция обработки одного элемента данных"""
    data, max_texts, counter, lock = args
    coll(data)
    with lock:
        counter.value += 1
    return 1

def main():
    max_texts = MAX_DATA  # Примерное значение
    num_workers = int(float(multiprocessing.cpu_count()) / 4)  # Количество процессов
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)  # Общий счетчик
    lock = manager.Lock()  # Блокировка для потокобезопасного увеличения счетчика
    pbar = tqdm(total=max_texts)

    with multiprocessing.Pool(processes=num_workers) as pool:
        args_list = [(str(data), max_texts, counter, lock) for data in DATA['text']]

        for result in pool.imap_unordered(process_data, args_list):
            if result == 1:
                pbar.update(1)
            if counter.value >= max_texts:
                break

    coll.save('final_save_texts_en')

    pbar.close()

def main_straight():
    for i, text in tqdm(enumerate(DATA['text'])):
        coll(text)
        if i > MAX_DATA:
            break
    
    coll.save('final_save')

if __name__ == "__main__":
    main()
    # main_straight()
