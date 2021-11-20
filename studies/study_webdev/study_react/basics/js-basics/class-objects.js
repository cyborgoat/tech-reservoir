// ES6
class Person {
    constructor() {
        this.name = "Robert";
        this.age = 23;
    }

    printMyName() {
        console.log(this.name);
    }
}

const person = new Person();
person.printMyName();

// ES7

class NewPerson {
    gender = 'male';
}