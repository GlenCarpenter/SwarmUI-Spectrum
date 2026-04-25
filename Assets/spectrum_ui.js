/** Resets all Spectrum group parameters to their registered defaults, without touching the group toggle. */
function spectrumResetDefaults() {
    for (let param of gen_param_types) {
        let group = param.group;
        while (group) {
            if (group.id == 'spectrum') {
                let elem = document.getElementById(`input_${param.id}`);
                if (elem) {
                    setDirectParamValue(param, param.default, elem, false, true);
                }
                break;
            }
            group = group.parent;
        }
    }
}

postParamBuildSteps.push(() => {
    let targetGroup = document.getElementById('input_group_content_spectrum');
    if (targetGroup) {
        targetGroup.append(createDiv('spectrum_reset_defaults_button', 'keep_group_visible',
            `<button class="basic-button" onclick="spectrumResetDefaults()">Reset Spectrum to Defaults</button>`));
    }
});
