stage=0
nbrun=1
models=`cat models.txt | grep -v "#" | tr "\n" " "`


# Download datasets
if [ $stage -le 0 ]
then
    python download_datasets_locally.py
fi

# Download models
if [ $stage -le 1 ]
then
    python download_models_locally.py
fi


#Corpus CAS
if [ $stage -le 2 ]
then

    #POS
    for model_name in $models
    do
        echo ${model_name}
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/cas/scripts/
            ./run_task_1.sh ${model_name}
            popd
        done
    done

    #CLS 
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/cas/scripts/
            ./run_task_2.sh ${model_name}
            popd
        done
    done

    #NER NEG
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            cd recipes/cas/scripts/
            ./run_task_3.sh ${model_name}
            cd ../../../
        done
    done

    #NER SPEC
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/cas/scripts/
            ./run_task_4.sh ${model_name}
            popd
        done
    done

fi


#Corpus CLISTER
if [ $stage -le 3 ]
then

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/clister/scripts/
            ./run.sh ${model_name}
            popd
        done
    done

fi



#CORPUS Diamed
if [ $stage -le 4 ]
then

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/diamed/scripts/
            ./run.sh ${model_name}
            popd
        done
    done

fi



#Corpus E3C
if [ $stage -le 5 ]
then

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/e3c/scripts/
            ./run.sh ${model_name} French_clinical
            popd
        done
    done

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/e3c/scripts/
            ./run.sh ${model_name} French_temporal
            popd
        done
    done

fi


#Corpus ESSAI
if [ $stage -le 6 ]
then

    #POS
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/essai/scripts/
            ./run_task_1.sh ${model_name}
            popd
        done
    done

    #CLS 
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/essai/scripts/
            ./run_task_2.sh ${model_name}
            popd
        done
    done

    #NER NEG
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            cd recipes/essai/scripts/
            ./run_task_3.sh ${model_name}
            cd ../../../
        done
    done

    #NER SPEC
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/essai/scripts/
            ./run_task_4.sh ${model_name}
            popd
        done
    done

fi



#Corpus FrenchMedMCQA
if [ $stage -le 7 ]
then

    #MCQA
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/frenchmedmcqa/scripts/
            ./run_task_1.sh ${model_name}
            popd
        done
    done

    #CLS
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/frenchmedmcqa/scripts/
            ./run_task_2.sh ${model_name}
            popd
        done
    done

fi


#Corpus MantraGSC
if [ $stage -le 8 ]
then

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/mantragsc/scripts/
            ./run.sh ${model_name} fr_emea
            popd
        done
    done

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/mantragsc/scripts/
            ./run.sh ${model_name} fr_medline
            popd
        done
    done

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/mantragsc/scripts/
            ./run.sh ${model_name} fr_patents
            popd
        done
    done

fi



#Corpus Morfitt
if [ $stage -le 9 ]
then

    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/morfitt/scripts/
            ./run.sh ${model_name}
            popd
        done
    done

fi

#Corpus PXCorpus
if [ $stage -le 10 ]
then

    #NER
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/pxcorpus/scripts/
            ./run_task_1.sh ${model_name}
            popd
        done
    done

    #CLS
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/pxcorpus/scripts/
            ./run_task_2.sh ${model_name}
            popd
        done
    done

fi


#Corpus QUAERO
if [ $stage -le 11 ]
then

    #EMEA
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/quaero/scripts/
            ./run.sh ${model_name} emea
            popd
        done
    done

    #MEDLINE
    for model_name in $models
    do
        for iteration in `seq 1 1 $nbrun`
        do
            pushd recipes/quaero/scripts/
            ./run.sh ${model_name} medline
            popd
        done
    done

fi

