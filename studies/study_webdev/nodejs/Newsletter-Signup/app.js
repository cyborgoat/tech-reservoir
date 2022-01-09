const express = require("express");
const bodyParser = require("body-parser");
const path = require("path");
const request = require("request");
const https = require("https");

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

    var data = {
        members: [
            {
                email_address: email,
                status: "subscribed",
                merge_fields: {
                    FNAME: firstName,
                    LNAME: lastName
                }
            }
        ]
    };

    var jsonData = JSON.stringify(data);

    const url = "https://us20.api.mailchimp.com/3.0/lists/d77fe71ad4";

    const options = {
        method: "POST",
        auth: "cyborgoat1:006cd6757ca1376022d2e3e3db3b6244-us20"
    }

    const request = https.request(url, options, function (response) {
        response.on("data", function (data) {
            console.log(JSON.parse(data));
        })
    });

    request.write(jsonData);
    request.end();
})


app.listen(3000, function () {
    console.log("Server is running on port 3000...");
});

//API Key
// 006cd6757ca1376022d2e3e3db3b6244-us20
//List ID
// d77fe71ad4