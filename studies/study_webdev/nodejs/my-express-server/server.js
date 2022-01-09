//jshint esversion:6

const express=require("express");
const app = express();

app.get("/",function (req,res) {
    console.log(req);
    res.send("Hello!")
});

app.get("/contact",function (req,res){
    res.send("Contact me at: cyborgoat@outlook.com");
})


app.listen(3000, function () {
    console.log("server started on port 3000...");
});
