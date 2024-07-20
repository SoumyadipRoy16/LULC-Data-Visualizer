document.addEventListener("DOMContentLoaded", function() {
    var legendContainer = document.getElementById("legend");
    {% if legend_data %}
        changeimagesource();
        legendContainer.style.display = "block";
    {% endif %}
});

function changeimagesource() {
    document.getElementById("mapImage").style.display = "none";
    document.getElementById("mapImage").style.display = "inline";
}

function validateForm() {
    var selectedYear = document.getElementById("selected_year").value;
    var selectedStateElement = document.getElementById("state_input");
    var selectedState = selectedStateElement ? selectedStateElement.value : '';
    var mandatoryMessage = document.getElementById("mandatoryMessage");

    if (selectedYear === "" || selectedState === "") {
        mandatoryMessage.style.display = "block";
        return false;
    } else {
        mandatoryMessage.style.display = "none";
    }

    document.getElementById("frm_lulc").submit();
}

// Adding event listeners to hide mandatory message on dropdown selection
document.getElementById("selected_year").addEventListener("change", function() {
    document.getElementById("mandatoryMessage").style.display = "none";
});

document.getElementById("state_input").addEventListener("change", function() {
    document.getElementById("mandatoryMessage").style.display = "none";
});