$("h1").addClass("big-title margin-50")
// $("h1").text("GoodBye")
// $("h1").removeClass("big-title")


console.log($("img").attr("src"));

$("a").attr("href", "https://www.baidu.com")

// Add event listener

$("h1").click(function () {
    $("h1").css("color", "blue");
});

// for (var i = 0; i < 5; i++) {
//     document.querySelectorAll("button")[i].addEventListener("click", function () {
//         document.querySelector("h1").style.color = "purple";
//     })
// }

$("button").click(function () {
    $("h1").css("color", "purple");
    // $("button").text("Dont Click Me");
    $("button").html("<em>Bye</em>");
})

///
$("input").keydown(function (event) {
    console.log(event.key)

})

// actions
$(document).keydown(function (event) {
    console.log(event.key);
    $("h1").text(event.key);

})

$("h1").on("mouseover", function (event) {
    $("h1").css("color", "green");
})

// add & remove
$("h1").before("<button>New</button>");
$("h1").after("<button>New</button>");
$("h1").prepend("<button>New</button>");
$("h1").append("<button>New</button>");
// $("button").remove();

// Animation1
// $("button").on("click", function () {
//     // $("h1").hide();
//     $("h1").toggle();
//     // $("h1").fadeIn();
//     // $("h1").fadeOut();
//     $("h1").fadeToggle();
// });
// // Animation2
// $("button").on("click", function () {
//     $("h1").animate({
//         opacity: 0.5,
//         margin: "20%",
//     })
// });
// Animation3
$("button").on("click", function () {
    $("h1").slideUp().slideDown().animate({
        opacity: 0.5,
        margin: "20%",
    })
});
