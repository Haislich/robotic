# Intent

The user's intentions for the library, in the user's own words where possible. This document is owned by the user and changes only with their approval. Every design decision cites the intention it serves (I1, I2, ...); reviews check the design against this document, not only the code against the design.

Quotes are the user's, from the review session of 2026-10-26, with typing slips corrected.

## I0. Wrong situations are unrepresentable

The guiding principle, as in Rust: strict typing on the symbolic side makes wrong situations unrepresentable. Two 3x3 matrices outside SO(3), or a rotation composed with the wrong frame, cannot be expressed. The numeric side is generated from the symbolic one, so it is correct by construction: it is always well defined and gives the expected result. When something goes wrong, the mathematical model is known to be sound and the problem lies elsewhere.

> "Se da qualche parte sbaglio a mettere un numero o un valore ho due 3x3 non di SO(3): JAX è felice di sommarle e moltiplicarle ignorando tutte le regole di SO(3). Per costruzione la simbolica si impegna a rendere inesprimibile una situazione del genere, quindi il JAX generato non può avere dei problemi, è sempre ben definito e dà il risultato atteso."

> "Immaginalo come il mantra di Rust: la simbolica, attraverso tipizzazione stretta, ci permette di rendere delle situazioni errate irrappresentabili, rendendo per costruzione verificabile la parte numerica. Così so in ogni momento che se c'è qualcosa che non va, il modello matematico è sound e il problema sta da qualche altra parte."

## I1. Purpose

The library exists to understand robotics by writing the mathematics the way an exam solution writes it. The current work builds the base: an architecture to stabilize and validate before Euler angles, DH tables and Jacobians are built on it.

> "Non pretendo che in questo momento sia tutto implementato, ed è il motivo per cui Eulero è deferred, ma mi interessa avere un POC per l'architettura che possiamo validare."

## I2. The contract is robotics mathematics with its meaning in the types

A value says what it is and with respect to what: ^A R_B is the rotation of B relative to A, not an orthonormal matrix; a vector is not three numbers. Reading or writing the code, it is clear which frame everything belongs to, also when coming back to an old exam much later.

The API guides: it forbids operations that are mathematically valid but meaningless in robotics. The type checker is the oracle, because it reports mistakes in the editor while the code is written. The mathematics is meaningful within the perimeter the types can describe.

> "La matematica è felice di accettare una matrice ortonormale di numpy, ma semanticamente per me non ha alcun senso, non risponde a domande tipo: rotazioni di chi, rispetto a chi. [...] Possono essere patate o mele, l'API deve fare in modo che sia nello scrivere che nel leggere il codice sia chiaro che sono mele o patate."

> "^A R_B ^A R_C ha comunque senso matematicamente, perché sono dei numeri/simboli, ma starei facendo un'operazione non permessa nel contesto della semantica associata a quelle matrici e l'API ha proprio il compito di guidarmi."

## I3. Two lines of defense

Static checks are the first line. Python's types are hints, so every static promise must also be enforced at runtime: a mistake the type checker rejects, if ignored, should raise at the operation that commits it, with a message naming what is wrong. If the type checking is done correctly, we have all the information available to give good diagnostics. It never produces a result that fails much later without indication. Where Python's types cannot express a rule, the runtime check is the only line. The two coexist; a runtime check never replaces a static guarantee that can be had and verified.

> "Mi sarei aspettato che un errore statico, se ignorato, causi un errore di runtime. Il problema degli errori di runtime è che spesso le condizioni che li causano si verificano dopo molto tempo e senza alcuna indicazione precisa."

> "Il modello dei tipi di Python è volutamente limitato, quindi ci sono molte cose direttamente non esprimibili e testabili a livello statico; per quelle situazioni l'approccio dinamico a runtime è l'unica soluzione."

> "Volevo che coesistessero, ma che non fosse un modo per abusare del runtime se possiamo avere delle garanzie statiche e verificabili."

## I4. The contract grows by addition; the implementation is free

The contract is designed for future additions: a new feature extends it without overturning it, and fixing a bug or replacing the implementation never changes it. Not every use case is known yet, so a past decision may need revisiting when a new use shows it does not fit; then the contract may change, with the user's approval. Such changes are the exception, not the norm. What is not yet decided fails loudly, so that deciding it later adds to the contract instead of changing it.

> "La regola generale è che l'API deve essere pensata per aggiunte future, ma tale per cui non dobbiamo stravolgerla a ogni nuova aggiunta. [...] Ci saranno situazioni in cui delle decisioni prese in passato richiederanno rivisitazioni; in quei casi è possibile modificare l'API, ma dobbiamo cercare di rendere questi casi l'eccezione, non la norma."

> "Immagina di avere una struttura dati [...] e usi come backend una lista. [...] Se nel corso d'opera esce fuori un requisito per cui i doppioni vanno filtrati, puoi scambiare la struttura sottostante per un insieme. L'API rimane la stessa ed è fissata, quello che cambia è il comportamento sottostante."

> "Scegliamo una struttura che non necessiti continui ritocchi, in modo che quello che costruiamo sulla struttura non abbia effetto su di essa."

A decision whose direction is not yet clear is not forced. It is recorded as open, and the contract promises nothing about it, so any later answer is an addition.

> "Non so e non voglio risponderti adesso, perché non ho chiaro in mente la direzione dell'API e rischio di dare una risposta che poi mi si ritorce contro. [...] Se questa decisione poi ha un'influenza sul futuro è un problema; quando ti parlavo di libertà intendevo proprio questo."

## I5. The wrapper hides the tools

sympy and JAX are implementation details. Their idiosyncrasies, costs and failure modes are hidden inside the library and never reach the contract: knowing that one sympy operation is slower than another teaches nothing about robotics. These details are Claude's responsibility.

> "L'utente finale non si deve preoccupare di dettagli di implementazione o di costi, devono essere nascosti nel wrapper. [...] Sono informazioni che danneggiano lo studente perché non aggiungono nulla sulla matematica e aggiungono una difficoltà dovuta ai mezzi usati."

> "Le idiosincrasie di entrambi sono dettagli implementativi di cui non ho tempo né voglia di occuparmi e per cui mi aspetto un aiuto concreto."

## I6. Symbolic and numeric have distinct jobs

The symbolic side is for study and is used directly. The numeric side is for simulation, an activity of a different nature. Neither is secondary. The cost of strict typing and semantics is paid entirely on the symbolic side, so the numeric side is unconstrained: a compiled function is indistinguishable from the same function written by hand in JAX, and it is correct by construction (I0). The numeric side reuses the semantics the symbolic side already builds, instead of rewriting it. Making the two coexist is a central design problem, not an implementation detail. The symbolic side has no time limit; its cost is paid once, when a model is built and compiled.

> "Il numerico sembra un afterthought ma non ha meno importanza, è che assolve un compito differente; la meno importanza è dovuta alla tipizzazione stretta e alla semantica associata nei tipi: l'overhead è localizzato tutto nella parte simbolica così da avere una parte numerica senza vincoli."

> "Per la simulazione JAX puro sarebbe stato abbastanza, ma se nella parte simbolica abbiamo già messo tutto in piedi per avere una semantica più pulita, perché non utilizzarlo?"

> "La divisione delle due cose è netta proprio perché pagare il costo anche nella parte numerica è inaccettabile."

## I7. Verbosity only when nothing else is as explicit

Verbosity is accepted when no other way makes the mathematics equally explicit. A shorter spelling is welcome only if it loses no expressiveness.

> "Se si potesse migliorare mi farebbe piacere, ma non a scapito dell'espressibilità."

## I8. Symbolic values are undetermined until their values are fixed

Equality and zero have meaning only where the values are numerically defined. Until then, the symbolic side is in a normal, honest indeterminacy: it does not decide whether something is zero or whether two things are equal, because no proof method is guaranteed in every case, and an answer that is "probably true until proven otherwise" can fail at the worst moment, during an exam.

> "L'uguaglianza ha senso solo nel contesto in cui i valori sono numericamente definiti. Fintanto che io non fisso i valori mi trovo senza mezzi termini in una situazione di indeterminazione normale e onesta."

Recorded 2026-10-27, after the code review of review round 4.

## Declared limits

- Scalars are dimensionless: angles and lengths are both scalars; the type does not tell them apart. This limit is permanent, because introducing quantity types later would change the contract.

> "Lcos(θ) è una cosa che viene usata sempre, ma cosa ritorna cos(θ)? Potremmo spendere ore a pensare a questa cosa, ma il ritorno diventa marginale e toglie importanza ai veri concetti chiave."

- Complex values are not supported: a computation that produces one (the square root of a negative constant) is outside the contract, and what follows is not promised.

## Open on purpose

- Singular configurations in compiled code (a normalized vector or an axis evaluated where its norm vanishes): whether they raise, carry a validity flag, or are caught by a debug mode. The contract promises no behavior here until the use is clear.

## Deliberately outside the contract

Printed form of results, optimization of the symbolic side, visualization in notebooks. They are expected to emerge from use once the contract is stable.
