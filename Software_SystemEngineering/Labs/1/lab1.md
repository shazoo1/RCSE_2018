## Use-cases

#### Use-case specification
* Use Case Name: Schedule the timetable
* Actors: Chef(primary), Provider(secondary)
* Summary Description: Update the available films list
* Pre-Conditions:
  - Already signed contract
* Post-Conditions:
  - Organize a new film showing
* Basic path:
  - Order a new set of films
  - Arrange the film delivery
  - Approve the delivery
  - Order an schedule updating


#### Use-case specification
* Use Case Name: Organize a film showing
* Actors: Tech Staff(primary), Cashier(primary)
* Summary Description: Prepare the cinema for a new film to be shown
* Pre-Conditions:
  -  Timetable is rescheduled
* Post-Conditions:
  - Set up an environment
  - Update promo lists
* Basic path:
  - Await for a new schedule to be approved and ordered to be executed
  - Prepare technical equipment
  - Update the promo materials


#### Use-case specification
* Use Case Name: Buy ticket
* Actors: Client(primary), Cashier(secondary)
* Summary Description: A way to take a ticket for a film session
* Post-Conditions:
  -  Calculate tips
* Basic Path:
  - Come at ticket office
  - Exchange money for a ticket
  - Store money in cashbox

#### Use-case specification
* Use Case Name: Take Salary
* Actors: Cashier(primary), Tech Staff(primary), Chef(secondary)
* Pre-Conditions:
  - Calculate tips
* Basic path:
  - Calculate extra payment for sellings
  - Approve final  the final salary
  - Receive the payment
