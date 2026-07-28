params.each { k, v -> 
    println "- $k = $v"
}
println "=============================\n"

workflow {
    // preprocess FLUXNET
    prep_fluxnet = Preprocess_FLUXNET()

    // download MODIS images at fluxnet sites
    modis_fluxnet = Download_MODIS_fluxnet(prep_fluxnet)

    // preprocess TEM and reanalysis data
    prep_tem = Preprocess_TEM()

    // download MODIS images for 0.5 degree grid cells
    ch_a = Channel.of( 1..40 )
    modis_tem = Download_MODIS_global(prep_tem, ch_a)
    modis_tem = modis_tem.collect()

    // preprocess global MODIS images
    ch_b = Channel.of( 1..100 )
    prep_modis = Prep_MODIS_global_1(modis_tem, ch_b)
    prep_modis = prep_modis.collect()
    prep_modis = Prep_MODIS_global_2(prep_modis)
    ch_c = Channel.of( 1..100 )
    prep_modis = Prep_MODIS_global_3(prep_modis, ch_c)
    prep_modis = prep_modis.collect()
    modis_prepped = Prep_MODIS_global_4(prep_modis)

    // preprocess specific to model
    prep_model = Preprocess_model(modis_fluxnet, prep_tem, modis_prepped)

    // train
    ch_d = Channel.of( 1..(params.num_sites) )
    ch_e = Channel.of( 1..(params.num_reps) )
    combined_channel = ch_d.combine(ch_e)
    trained = Train(prep_model, combined_channel)
    trained = trained.collect()

    // evaluate
    test = Eval(trained)

    // preprocess for upscaling (using separate wetland map)
    prep_upscale_wad2m = Preprocess_upscale_WAD2M(test)

    // train for upscaling (nothing held-out)
    ch_f = Channel.of( 1..(params.num_reps) )
    trained_upscale_wad2m = Upscale_train_WAD2M(prep_upscale_wad2m, ch_f)
    trained_upscale_wad2m = trained_upscale_wad2m.collect()

    // predict every grid cell
    ch_g = Channel.of( 1..100 )    
    upscaled_wad2m = Upscale_WAD2M(trained_upscale_wad2m, ch_g)
    upscaled_wad2m = upscaled_wad2m.collect()

    // final upscaling analysis and plots
    Global_plot_WAD2m(upscaled_wad2m)    
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



process Download_MODIS_fluxnet {
    // resources
    memory '4 GB'
    time '1h'

    // misc. settings
    tag "download_modis_fluxnet"
    conda "${params.repo}/requirements.yml"

    output:
    path "modis_images_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/gee_pulldown_FLUXNET.py
    echo "Done." > modis_images_done.txt
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



process Download_MODIS_global {
    // resources
    memory '20 GB'
    time '72h'

    // misc. settings
    tag "download_modis_fluxnet_${rep}"
    conda "${params.repo}/requirements.yml"

    intput:
    tuple path "prep_TEM.sav", int(rep)

    output: path "modis_images_done_${rep}.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/gee_pulldown_global.py ${rep}
    echo "Done." > modis_images_done_${rep}.txt
    """
}



process Prep_MODIS_global_1 {
    // resources
    memory '4 GB'
    time '24h'

    // misc. settings
    tag "prep_modis_global_1_${rep}"
    conda "${params.repo}/requirements.yml"

    input: int(rep)

    output: path "modis_prep1_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step1.py ${rep}
    echo "Done." > modis_prep1_done_${rep}.txt
    """
}



process Prep_MODIS_global_2 {
    // resources
    memory '200 GB'
    time '2h'

    // misc. settings
    tag "prep_modis_global_2"
    conda "${params.repo}/requirements.yml"

    output: path "modis_prep2_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step2.py
    echo "Done." > modis_prep2_done.txt
    """
}




process Prep_MODIS_global_3 {
    // resources
    memory '200 GB'
    time '1h'

    // misc. settings
    tag "prep_modis_global_3_${rep}"
    conda "${params.repo}/requirements.yml"

    input: int(rep)

    output: path "modis_prep3_done.txt"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step3.py ${rep}
    echo "Done." > modis_prep3_done_${rep}.txt
    """
}



process Prep_MODIS_global_4 {
    // resources
    memory '200 GB'
    time '1h'

    // misc. settings
    publishDir "${params.workdir}/Out/MODIS_tiles_TEM/Preprocessed_tiles/", mode: 'copy'
    tag "prep_modis_global_4"
    conda "${params.repo}/requirements.yml"

    output: path "global_SDs.npy"

    script:
    """
    python ${params.repo}/Code/Google_earth_engine/prep_MODIS_step4.py ${rep}
    echo "Done." > modis_prep4_done_${rep}.txt
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
    tuple path "modis_images_done.txt", path "prep_TEM.sav", path "prep_obs.sav"

    output:
    path "prep_model.sav"

    script:
    """
    python ${params.repo}/Code/preprocess_ml_cnn.py
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



process Preprocess_upscale_WAD2M {
    // resources
    memory '200 GB'
    time '4h'

    // misc. settings
    publishDir "${params.workdir}/Out/Upscale_WAD2M/", mode: 'copy'
    tag "preprocess_upscale_wad2m"
    conda "${params.repo}/requirements.yml"

    output:
    path "prep_upscale_WAD2M.sav"

    script:
    """
    python ${params.repo}/Code/${params.model_version}/preprocess_upscale_WAD2M.py
    """
}



process Upscale_train_WAD2M {
    // resources
    memory '50 GB'
    time '1h'

    // misc. settings
    publishDir "${params.workdir}/Out/Upscale_WAD2M/", mode: 'copy'
    tag "train_upscale_${rep}"
    conda "${params.repo}/requirements.yml"

    input:
    tuple int(rep)

    output:
    path "production_rep_${rep}.sav"

    script:
    """
    python ${params.repo}/Code/${params.model_version}/train.py \
        0 \
        ${rep}
    """
}




process Upscale_WAD2M {
    // resources
    memory '50 GB'
    time '24h'

    // misc. settings
    publishDir "${params.workdir}/Out/Upscale_WAD2M/", mode: 'copy'
    tag "upscale_wad2m"
    conda "${params.repo}/requirements.yml"

    input: int(rep)

    output:
    path "upscale_wad2m_${rep}.txt"

    script:
    """
    python ${params.repo}/Code/${params.model_version}/upscale_WAD2M.py ${rep}
    echo "Done." > upscale_wad2m_${rep}.txt
    """
}



process Global_plot_WAD2m {
    // resources
    memory '200 GB'
    time '2h'

    // misc. settings
    publishDir "${params.workdir}/Out/Upscale_WAD2M/", mode: 'copy'
    tag "plot_upscale_wad2m"
    conda "${params.repo}/requirements.yml"

    output:
    path "hybrid.pdf"

    script:
    """
    python ${params.repo}/Code/${params.model_version}/plot.py
    """
}