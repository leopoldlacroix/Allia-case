We are running a technical test to do at home, to be developed in a few hours, to be handed over when you're happy with the result, ideally within a week (on a private Git repository, by inviting jguepin@getallia.com and ebazarov@getallia.com or jguepin and EBazarov on GitHub).
The overall idea: 
Accountants receive dozens of emails daily, we want to handle emails from clients and automatically generate answers to make their life and inbox management easier. 
First, we want to classify emails, to understand: 
if the email is from a client, an internal email, or an external email (marketing, newsletter, spam…)
the topic of the email (accounting, taxes, payroll, legal, other)
The goal of the project is to build an agent or set of rules that will analyse an email provided as input, and provide a classification to answer these questions as an output.
The attached JSON contains a list of examples of emails, with the expected input format.
You can use Python and any library you need. 
 
What we're going to look at:
The methodology used to answer the problem.
The quality of the code, if it is easy to read, understand and well organized.
The relevance of the result, not necessarily 100% success, but a minimum of false positives.
