import numpy as np


SEQUENCE_LENGTH = 8
MAX_USER_ID = 19438

CATEGORIES = ['Restaurants',
 'Food',
 'Shopping',
 'Nightlife',
 'Beauty & Spas',
 'Bars',
 'Home Services',
 'Local Services',
 'Health & Medical',
 'Sandwiches',
 'Coffee & Tea',
 'Event Planning & Services',
 'Pizza',
 'American (New)',
 'Breakfast & Brunch',
 'Automotive',
 'American (Traditional)',
 'Active Life',
 'Arts & Entertainment',
 'Fashion',
 'Hair Salons',
 'Italian',
 'Specialty Food',
 'Chinese',
 'Nail Salons']


EMPTY_EMBEDDING = np.array([-1.26946613e-01,  3.12528796e-02, -1.02809288e-01,  2.44173082e-03,
       -4.75614481e-02,  6.17677383e-02,  2.29772297e-03, -5.92133701e-02,
       -9.21589658e-02,  2.96858372e-03,  9.41385608e-03,  5.65042347e-02,
       -1.31870788e-02, -1.03005609e-02, -5.60650527e-02,  5.92284277e-02,
        3.08383554e-02, -1.00667976e-01, -5.97230508e-04, -2.91800033e-02,
       -7.38585666e-02,  1.06021157e-02,  1.11927055e-01,  1.24340914e-02,
       -1.00755252e-01,  3.27105671e-02, -3.03749498e-02,  6.91077262e-02,
       -4.19381671e-02, -6.44306540e-02, -7.20828027e-02,  1.14573076e-01,
       -6.46825209e-02,  5.72248064e-02, -5.55877835e-02, -2.22194027e-02,
        7.11174533e-02,  3.58592682e-02,  2.69906744e-02,  1.43543938e-02,
       -5.65335974e-02,  3.62016186e-02,  4.56811674e-02,  2.42449343e-02,
        1.78285071e-03,  4.42870632e-02, -6.30023330e-02, -9.48341787e-02,
        5.88914379e-03,  7.89088905e-02,  7.92749375e-02, -2.47562863e-02,
        6.21500090e-02, -1.33969307e-01, -5.38650379e-02,  5.71366251e-02,
       -4.64430898e-02, -1.14143431e-01,  1.55266607e-02, -1.08681202e-01,
        8.19185674e-02,  3.38046961e-02, -4.73130234e-02,  5.44593669e-04,
        4.83158454e-02, -1.96418799e-02, -4.32654656e-02, -7.87212700e-02,
       -8.51582512e-02, -1.70528553e-02, -1.45261153e-03,  1.37175098e-02,
        4.96391095e-02, -1.26424022e-02,  4.23519425e-02,  8.01903307e-02,
        7.31598772e-03, -1.62489899e-02, -1.86917782e-02,  2.31401576e-03,
        7.76172727e-02, -2.56951489e-02, -2.51028799e-02, -4.78181019e-02,
        3.03840358e-03, -1.30811250e-02,  5.94434105e-02, -7.21288547e-02,
       -4.72790971e-02, -3.19236480e-02,  4.62397821e-02,  7.54106268e-02,
       -1.84193458e-02,  4.38040160e-02, -3.50319259e-02,  6.88391104e-02,
       -3.19954455e-02,  6.85291132e-04,  7.47733191e-03,  5.17894365e-02,
        6.31876588e-02,  5.30797578e-02,  3.99430133e-02, -7.84801990e-02,
       -2.64160465e-02, -1.64072923e-02, -2.75619607e-02,  5.74342534e-02,
        6.00629998e-03,  3.32059059e-03,  4.69495691e-02,  6.69591427e-02,
       -2.35086698e-02,  5.11521548e-02, -1.70836262e-02,  1.16045296e-01,
        9.37262699e-02,  4.52567786e-02,  1.57890618e-02,  2.84157172e-02,
        3.14323939e-02, -2.83792168e-02, -4.15497310e-02, -5.70224300e-02,
        1.39704114e-02,  5.01492023e-02,  4.11274284e-02, -4.81174433e-33,
       -1.14744538e-02, -2.02196557e-02, -2.35037580e-02,  2.42869146e-02,
        6.45171031e-02, -2.49791574e-02, -5.43496422e-02, -6.37661619e-03,
       -1.37188032e-01, -3.03322021e-02,  5.46212308e-02, -6.93802834e-02,
       -4.25976980e-03,  5.80615327e-02, -4.85774651e-02, -2.29473133e-02,
       -3.27735208e-02, -2.02285983e-02, -2.72103213e-02, -4.68579528e-04,
       -4.85672839e-02,  6.34028390e-02,  3.15619819e-02, -5.57803698e-02,
       -3.38411238e-03,  1.40427335e-04, -1.77477207e-02,  6.92178030e-03,
        6.05836846e-02,  5.86200655e-02, -1.54051021e-01,  2.22251676e-02,
       -2.63652243e-02,  2.48088920e-03, -2.11084113e-02,  1.95291284e-02,
       -5.20094037e-02,  2.16142703e-02,  2.18960308e-02, -3.53022180e-02,
        9.78825707e-03, -4.43639532e-02, -7.39443675e-02, -4.75575496e-03,
       -4.47773524e-02,  8.79632533e-02, -3.00051412e-03,  2.50088107e-02,
        2.53385417e-02,  5.12641259e-02, -6.91960827e-02, -2.90935542e-02,
        5.02461428e-03,  1.02569886e-01, -6.40913285e-03, -1.91082973e-02,
        2.65642479e-02,  5.39383944e-03,  1.06040109e-02,  4.31339368e-02,
        5.84442988e-02,  9.62109938e-02, -2.25387234e-03, -5.85674047e-02,
       -5.07967882e-02, -4.98541864e-03, -1.46457208e-02,  2.64195353e-02,
        4.94117774e-02, -3.54656540e-02, -5.64341694e-02,  2.45988388e-02,
        6.56523034e-02, -6.25024140e-02,  2.30239704e-02, -1.08286915e-02,
       -2.62614712e-02, -2.18742210e-02, -2.04494800e-02, -1.11175915e-02,
        5.93754984e-02, -4.87286709e-02, -5.49878217e-02, -4.79501262e-02,
       -1.02680102e-01, -3.30601893e-02,  7.80179873e-02, -7.34903524e-03,
        1.83219798e-02,  1.90431275e-03, -6.28508925e-02,  5.15789837e-02,
        1.07693449e-01, -3.98199446e-02, -1.06440885e-02,  3.16077317e-33,
        1.68458875e-02, -6.25446141e-02,  1.05401101e-02,  1.35018267e-02,
       -4.54171114e-02, -1.69595666e-02, -8.07032958e-02,  1.26907513e-01,
        2.98538078e-02,  6.23291135e-02,  4.66952436e-02, -8.77641067e-02,
        2.73072682e-02, -1.93957873e-02,  2.84570120e-02, -4.63604107e-02,
        1.86879409e-03, -4.65013757e-02,  5.34766540e-02, -1.88846942e-02,
        3.57384421e-02,  5.23607694e-02,  6.48911635e-04, -7.00172037e-02,
       -4.23809476e-02,  8.89529213e-02,  4.54587974e-02,  1.77005697e-02,
        2.15554778e-02, -2.59693116e-02, -1.04155783e-02, -2.12136749e-02,
       -2.46889214e-03,  4.57849428e-02,  4.25228290e-02, -2.93562864e-03,
       -2.24905051e-02,  1.45582044e-02, -1.29488716e-02,  8.01740065e-02,
        1.85258631e-02,  3.65460105e-02,  4.61329632e-02, -6.39327168e-02,
       -5.59025556e-02, -2.64681354e-02,  7.24311695e-02,  5.39469421e-02,
       -1.30307879e-02,  2.71533355e-02, -3.13893445e-02, -4.07525850e-03,
       -5.05823195e-02, -7.53404126e-02, -3.08218114e-02, -1.03550993e-01,
       -9.96762048e-03, -2.89064217e-02,  3.87958512e-02, -1.28529472e-02,
       -3.24959382e-02,  8.84590968e-02, -3.29027548e-02, -2.52474919e-02,
        2.79008616e-02, -1.14601711e-02,  7.82860965e-02, -3.05488482e-02,
        2.53860336e-02,  1.04361631e-01,  4.25445149e-03,  7.18741789e-02,
        3.73404808e-02,  2.67072897e-02,  6.61541373e-02, -5.59313297e-02,
        1.26187474e-01, -6.60484582e-02,  2.84691006e-02,  8.08180030e-03,
        5.28714294e-03, -2.46844143e-02,  4.80422191e-02,  1.52891241e-02,
        7.95221031e-02, -2.34660171e-02,  2.83347699e-03, -1.51589308e-02,
       -1.99815556e-02,  1.61818359e-02, -2.50181258e-02,  3.79047953e-02,
       -2.06484664e-02, -2.75020096e-02,  3.88726704e-02, -1.83717752e-08,
        1.71381664e-02,  9.57573727e-02,  1.24591207e-02,  4.70145978e-02,
        1.48001388e-02, -6.40981868e-02,  4.44394313e-02,  8.64674896e-03,
       -4.97674048e-02,  1.63089093e-02,  3.55674997e-02,  8.84409156e-03,
        1.16836624e-02, -4.64679068e-03, -6.43802807e-02, -8.17686766e-02,
        3.77261117e-02,  1.73224539e-01, -3.65776345e-02, -4.09974344e-02,
        3.27829481e-03,  5.76814003e-02,  2.99111698e-02,  2.13383343e-02,
        8.05065408e-03, -2.19991077e-02,  4.60963100e-02, -6.67959452e-02,
       -2.71091498e-02, -1.27491690e-02,  2.31833812e-02,  1.13935106e-01,
       -3.37131955e-02, -1.39391730e-02, -5.79375252e-02,  5.61274365e-02,
       -1.99941313e-03,  2.97831129e-02,  7.61376992e-02, -4.65535782e-02,
       -2.60717440e-02, -1.63737714e-01, -3.77274156e-02, -1.34848012e-02,
       -2.18700841e-02,  3.31706628e-02, -1.03413183e-02, -3.68049145e-02,
        3.32469568e-02, -3.36191729e-02,  3.94219160e-02, -4.78863940e-02,
        1.74969416e-02, -2.30008494e-02,  8.63201693e-02,  4.18049702e-03,
       -2.13608406e-02, -2.23481543e-02, -1.05589516e-02,  9.04311761e-02,
        4.26763631e-02, -2.12164386e-03, -1.31416973e-03,  1.11585923e-01])
