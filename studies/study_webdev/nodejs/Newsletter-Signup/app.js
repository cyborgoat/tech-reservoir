const express = require("express");
const bodyParser = require("body-parser");
const path = require("path");
const request = require("request");

const app = express();

app.use(express.static("static"));
app.use(bodyParser.urlencoded({extended: true}));


app.get("/", function (req, res) {
    res.sendFile(path.join(__dirname, 'signup.html'));
});


app.post("/", function (req, res) {
    var firstName = req.body.fname;
    var lastName = req.body.lname;
    var email = req.body.email;
    console.log(firstName + lastName + email);
})


app.listen(3000, function () {
    console.log("Server is running on port 3000...");
});