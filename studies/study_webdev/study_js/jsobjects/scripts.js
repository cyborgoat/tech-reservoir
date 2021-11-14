var housekeeper = {
    yearsOfExperience: 12,
    name: "Jane",
    cleaningRepertoire: ["bathroom", "lobby", "bedroom"]
}

function BellBoy(name, age, hasWorkPermit, languages) {
    this.name = name;
    this.age = age;
    this.hasWorkPermit = hasWorkPermit;
    this.languages = languages;
    this.clean = function (){
        alert("Cleaning in progress...")
    }
}

// Initialize Object
var bellBoy1 = new BellBoy("Timmy", 19, true, ["french", "english"]);