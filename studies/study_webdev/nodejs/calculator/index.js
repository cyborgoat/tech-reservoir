const express = require("express");
const bodyParser = require("body-parser");

const app = express();
app.use(bodyParser.urlencoded({extended: true}));

app.get("/", function (req, res) {
    // res.send("<h1>Hellod MF~</h1>");
    res.sendFile(__dirname + "/index.html");
})

app.post("/index.html", function (req, res) {
    console.log(req.body);
    var num1 = req.body.num1;
    var num2 = req.body.num2;
    var result = num1 + num2;
    res.send("Thanks for posting that! The result is " + result);
})

app.listen(3000, function () {
    console.log("Server is listening on port 3000...");
});