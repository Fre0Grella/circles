#!/bin/sh

# Questo script richiede 5 parametri per essere eseguito:
# 1 - il nome del programma da eseguire
# 2 - numero di ripetizioni
# 3 - tipo di scaling (1 per prob. crescente, 0 per it. cresccente)
# 4 - grandezza del problema base
# 5 - numero di iterazioni base

# Questo script misura il tempo di esecuzione del programma scelto 
# aumentando la grandezza del problema. il numero di prove aumentando
# la grandezza del problema può essere cambiato modificando il valore
# REP2 presente qui sotto. 
#
# Questo script esegue il programma sfruttando il massimo numero di
# thread disponibili se eseguito con OpenMP.

# Ultimo aggiornamento 2023-02-06
# Marco Galeri (marco.galeri@studio.unibo.it)

PROG=$1
REP=$2
TYPE=$3
N0=$4   # base problem size
IT=$5   #iterations
REP2=10


if [ ! -f "$PROG" ]; then
    echo
    echo "Non trovo il programma $PROG."
    echo
    exit 1
fi

echo -n "p\t"

for t in `seq $REP`; do
echo -n "t$t\t"
done
echo ""

for p in `seq $REP2`; do
    
    N=`echo "$N0 * $p" | bc -l -q`;
    IT_SIZE=`echo "$IT * $p" | bc -l -q`
    if [ "$TYPE" = 1 ]; then
    echo -n "$N\t"
    else
    echo -n "$IT_SIZE\t"
    fi

    for rep in `seq $REP`; do
        if [ "$TYPE" = 1 ]; then
        EXEC_TIME="$("./"$PROG $N $IT | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        else
        EXEC_TIME="$("./"$PROG $N0 $IT_SIZE | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        fi
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
