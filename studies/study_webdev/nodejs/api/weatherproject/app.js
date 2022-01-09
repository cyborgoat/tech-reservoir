const express = require("express");
const https = require("https");
const {response} = require("express");
const app = express();

// Get Part
app.get("/", function (req, res) {
    const url = "https://api.openweathermap.org/data/2.5/weather?q=London&appid=99adfc4f31711bef920c481a47736267&units=metric";
    https.get(url, function (response) {
        console.log(response.statusCode);

        response.on("data", function (data) {
            const weatherData = JSON.parse(data);
            console.log(JSON.stringify(weatherData));
        });

    });


    res.send("Server is up and rudnning.")

})


// Listen Part
app.listen(3000, function () {
    console.log("Port listening on port 3000...");
})