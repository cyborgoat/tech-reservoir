const express = require("express");
const https = require("https");
const bodyParser = require("body-parser");
const app = express();
app.use(bodyParser.urlencoded({extended: true}));


// Get Part
app.get("/", function (req, res) {
    res.sendFile(__dirname + "/index.html");
    // res.send("Server is up and running.")
});

app.post("/", function (req, res) {
    const query = req.body.cityName;
    const apiKey = "99adfc4f31711bef920c481a47736267";
    const units = "metric";
    const url = "https://api.openweathermap.org/data/2.5/weather?q=" + query + "&appid=" + apiKey + "&units=" + units;

    https.get(url, function (response) {
        console.log(response.statusCode);

        response.on("data", function (data) {
            const weatherData = JSON.parse(data);
            // console.log(JSON.stringify(weatherData));

            const temp = weatherData.main.temp;
            const weatherDescription = weatherData.weather[0].description;
            const icon = weatherData.weather[0].icon;
            const imgUrl = "http://openweathermap.org/img/wn/" + icon + "@2x.png";
            // console.log(temp);
            res.write("<h1>The weather is currently " + weatherDescription + "</h1>");
            res.write("<h1>The temperature in " + query + " is " + temp + "degree Celsius.</h1>");
            res.write("<img src=" + imgUrl + ">");
            res.send();
        });

    });
});


// Listen Part
app.listen(3000, function () {
    console.log("Port listening on port 3000...");
})