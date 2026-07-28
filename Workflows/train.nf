params.each { k, v -> 
    println "- $k = $v"
}
println "=============================\n"

workflow {
    // preprocess FLUXNET
    prep_fluxnet = Preprocess_FLUXNET()

    // Preprocess TEM and reanalysis data
    prep_tem = Preprocess_TEM()

    // preprocess specific to model
    prep_model = Preprocess_model(prep_tem)

    // train
    ch_d = Channel.of( 1..(params.num_sites) )
    ch_e = Channel.of( 1..(params.num_reps) )
    combined_channel = ch_d.combine(ch_e)
    trained = Train(prep_model, combined_channel)
    trained = trained.collect()

    // evaluate
    test = Eval(trained)
}



process Preprocess_FLUXNET() {
    // resources
    memory '40 GB'
    time '1h'

    // misc. settings
    publishDir "${params.workdir}/Out/", mode: 'copy'
    tag "prep_fluxnet"
    conda "${params.repo}/requirements.yml"

    output:
    path "prep_obs.sav"

    script:
    """
    python ${params.repo}/Code/preprocess_fluxnet.py
    """
}



process Preprocess_TEM() {
    // resources
    memory '200 GB'
    time '4h'

    // misc. settings
    publishDir "${params.workdir}/Out/", mode: 'copy'
    tag "prep_TEM"
    conda "${params.repo}/requirements.yml"

    output:
    path "prep_TEM.sav"

    script:
    """
    python ${params.repo}/Code/preprocess_TEM.py
    """
}



process Preprocess_model() {
    // resources
    memory '50 GB'
    time '2h'

    // misc. settings
    publishDir "${params.workdir}/Out/", mode: 'copy'
    tag "prep_model"
    conda "${params.repo}/requirements.yml"

    input:
    tuple path "prep_TEM.sav", path "prep_obs.sav"

    output:
    path "prep_model.sav"

    script:
    """
    python ${params.repo}/Code/preprocess_ml.py
    """
}



process Train {
    // resources
    memory '50 GB'
    time '1h'

    // misc. settings
    publishDir "${params.workdir}/Out/${params.model_version}/", mode: 'copy'    
    tag "train_${test_index}_${rep}"
    conda "${params.repo}/requirements.yml"

    input:
    tuple int(test_index), int(rep)

    output:
    path "result_${test_index}_rep_${rep}.txt"

    script:
    """
    python ${params.repo}/Code/${params.model_version}/train.py \
        ${test_index} \
        ${rep}
    """
}



process Eval {
    // resources
    memory '4 GB'
    time '1h'

    // misc. settings
    publishDir "${params.workdir}/Out/${params.model_version}/", mode: 'copy'
    tag "eval"
    conda "${params.repo}/requirements.yml"

    output:
    path "evaluation.pdf"
    
    script:
    """
    python ${params.repo}/Code/evaluate.py
    """
}
