// var buttonList = document.querySelectorAll("button .drum");
var buttonList = document.querySelectorAll(".drum");
console.log(buttonList)
for (var i = 0; i < buttonList.length; i++) {
    buttonList[i].addEventListener("click", function () {
        // alert("I got clicked!")
        this.style.color = "white";
        var buttonInnerHtml = this.innerHTML;
        makeSound(buttonInnerHtml);
    })
}

// Detecting Keyboard Press
document.addEventListener('keydown', function (event) {
    makeSound(event.key);
})

function makeSound(key) {
    switch (key) {
        case "w":
            var audio = new Audio("sounds/tom-1.mp3");
            audio.play();
            break;
        case "a":
            var audio = new Audio("sounds/tom-2.mp3");
            audio.play();
            break;
        case "s":
            var audio = new Audio("sounds/tom-3.mp3");
            audio.play();
            break;
        case "d":
            var audio = new Audio("sounds/tom-4.mp3");
            audio.play();
            break;
        case "j":
            var audio = new Audio("sounds/snare.mp3");
            audio.play();
            break;
        case "k":
            var audio = new Audio("sounds/crash.mp3");
            audio.play();
            break;
        case "l":
            var audio = new Audio("sounds/kick-bass.mp3");
            audio.play();
            break;
        default:
            console.log("Error");
            break;

    }
}

// var audio = new Audio("sounds/crash.mp3");
// audio.play();