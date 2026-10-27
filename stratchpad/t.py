from dataclasses import dataclass

from robotic import Frame, FreeAxis, Rotation, Symbol


class B1(Frame): ...


class B2(Frame): ...


q = Symbol("q")
r = Rotation.about_z(q, target=B1, source=B2)
r_i = r.inverse()
r_a = Rotation.about_x(q, target=B2, source=B1)
r_ai = r_a.inverse()
monster = r @ r_a @ r @ r_a @ r_ai @ r_i @ r_ai @ r_i
print(monster.simplify())


@dataclass
class Joint:
    q: float


monster_n = monster.compile(Joint)
print(monster_n(Joint(1)))
print(monster_n(Joint(2)))
