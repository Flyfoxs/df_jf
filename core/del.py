from redlock import RedLockError


from core.feature import *
#from core.check import check_options
import fire
from core.db import *
from core.predict import validate

from core.merge_multiple_file import select_col

def read_file(base_file):
    base = pd.read_csv(base_file)
    return base
if __name__ == '__main__':
    # validate(3, 'lr', 'var048', 'up', 0)


    pass
