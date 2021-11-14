// var buttonList = document.querySelectorAll("button .drum");
var buttonList = document.querySelectorAll(".drum");
console.log(buttonList)
for (var i = 0; i < buttonList.length; i++) {
    buttonList[i].addEventListener("click", function () {
        // alert("I got clicked!")
        var audio = new Audio("sounds/crash.mp3");
        audio.play();
    })
}