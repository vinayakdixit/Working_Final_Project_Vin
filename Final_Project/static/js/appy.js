
$(document).ready(function() {
  $(window).keydown(function(event){
    if(event.keyCode == 13) {
      event.preventDefault();
      return false;
    }
  });
});

// selecting div element from results.html where the json.dump object containing results was stored. .text() access text from the div
var data = d3.select("#divdata").text();
// re-creating json object from the  "divdata" above so that it can be accessed here
var points = JSON.parse(data);
// accessing properties of "points" object defined above which hold the model results
var rValues = [points.SVC_Prediction, points.NB_Prediction, points.XG_Prediction, points.KNN_Prediction, points.LR_Prediction, points.RF_Prediction];
// Array to display the labels (model names)
var rNames = ["SVC_Prediction", "NB_Prediction", "XG_Prediction", "KNN_Prediction", "LR_Prediction", "RF_Prediction"];

// Looping through each data point of the "rValues" array object derived from "points" json object above
for(var i=0;i<rValues.length;i++){
d3.select("body").append("span").html(rNames[i]);
    var svg = d3.select("body").append("svg");
    svg.attr("width", "150px").attr("height", "150px");
       var circles = svg.selectAll("circle");
         circles.data(rValues[i])
         .enter()
         .append("circle")
         .attr("cx", 75)
         .attr("cy", 75)
         .attr("r", 50)
         .attr("stroke", "black")
         .attr("stroke-width", "5")
        .attr("fill", function(d) {
         if (d ==0){return "green";}
         else {return "red";}
        });
}