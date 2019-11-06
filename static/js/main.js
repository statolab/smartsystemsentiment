function toggle_display(output_index){
    bs = document.getElementById('button_display_' + output_index);
    bc = document.getElementById('button_collapse_' + output_index);
    oa = document.getElementById('output_all_' + output_index);
    ol = document.getElementById('output_less_' + output_index);
    if(oa.style.display == 'none' || bs.style.display == 'block'){
        bs.style.display = 'none';
        ol.style.display = 'none';
        oa.style.display = 'block';
        bc.style.display = 'block';
    }else if(oa.style.display == 'block' || bs.style.display == 'none'){
        bs.style.display = 'block';
        ol.style.display = 'block';
        oa.style.display = 'none';
        bc.style.display = 'none';
    }
}