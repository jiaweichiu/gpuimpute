set -e

wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
wget http://files.grouplens.org/datasets/movielens/ml-10m.zip
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

unzip -o ml-1m.zip
unzip -o ml-10m.zip
unzip -o ml-20m.zip

mv ml-10M100K ml-10m

HEADER=`head ml-20m/ratings.csv -n 1`

for D in ml-1m ml-10m; do
  echo "Converting $D"
  { echo "${HEADER}"; cat ${D}/ratings.dat | sed -e s/::/,/g; } > ${D}/ratings.csv
done

for D in ml-1m ml-10m ml-20m; do
  wc -l ${D}/ratings.csv
done

RNGSEED=42
for D in ml-1m ml-10m ml-20m; do
  echo "Generating sparse matrices ${D}"
  python gen.py ${D}/ratings.csv ${D} --rng $RNGSEED --testfrac 0.1
done