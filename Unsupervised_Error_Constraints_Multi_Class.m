function [c, ceq] = Unsupervised_Error_Constraints_Multi_Class(x,agreement_rates,divisor)

e1 = x(1);
e2 = x(2);
e3 = x(3);
e4 = x(4);
e5 = x(5);
e12 = x(6);
e13 = x(7);
e14 = x(8);
e15 = x(9);
e23 = x(10);
e24 = x(11);
e25 = x(12);
e34 = x(13);
e35 = x(14);
e45 = x(15);
e123 = x(16);
e124 = x(17);
e125 = x(18);
e134 = x(19);
e135 = x(20);
e145 = x(21);
e234 = x(22);
e235 = x(23);
e245 = x(24);
e345 = x(25);
e1234 = x(26);
e1235 = x(27);
e1245 = x(28);
e1345 = x(29);
e2345 = x(30);
e12345 = x(31);
p12 = x(32);
p13 = x(33);
p14 = x(34);
p15 = x(35);
p23 = x(36);
p24 = x(37);
p25 = x(38);
p34 = x(39);
p35 = x(40);
p45 = x(41);
p123 = x(42);
p124 = x(43);
p125 = x(44);
p134 = x(45);
p135 = x(46);
p145 = x(47);
p234 = x(48);
p235 = x(49);
p245 = x(50);
p345 = x(51);
p1234 = x(52);
p1235 = x(53);
p1245 = x(54);
p1345 = x(55);
p2345 = x(56);
p12345 = x(57);


a12 = agreement_rates(1)/divisor;
a13 = agreement_rates(2)/divisor;
a14 = agreement_rates(3)/divisor;
a15 = agreement_rates(4)/divisor;
a23 = agreement_rates(5)/divisor;
a24 = agreement_rates(6)/divisor;
a25 = agreement_rates(7)/divisor;
a34 = agreement_rates(8)/divisor;
a35 = agreement_rates(9)/divisor;
a45 = agreement_rates(10)/divisor;
a123 = agreement_rates(11)/divisor;
a124 = agreement_rates(12)/divisor;
a125 = agreement_rates(13)/divisor;
a134 = agreement_rates(14)/divisor;
a135 = agreement_rates(15)/divisor;
a145 = agreement_rates(16)/divisor;
a234 = agreement_rates(17)/divisor;
a235 = agreement_rates(18)/divisor;
a245 = agreement_rates(19)/divisor;
a345 = agreement_rates(20)/divisor;
a1234 = agreement_rates(21)/divisor;
a1235 = agreement_rates(22)/divisor;
a1245 = agreement_rates(23)/divisor;
a1345 = agreement_rates(24)/divisor;
a2345 = agreement_rates(25)/divisor;
a12345 = agreement_rates(26)/divisor;

ceq = [1 - e1 - e2 + e12 - a12 + p12,
     1 - e1 - e3 + e13 - a13 + p13,
     1 - e1 - e4 + e14 - a14 + p14,
     1 - e1 - e5 + e15 - a15 + p15,
     1 - e2 - e3 + e23 - a23 + p23,
     1 - e2 - e4 + e24 - a24 + p24,
     1 - e2 - e5 + e25 - a25 + p25,
     1 - e3 - e4 + e34 - a34 + p34,
     1 - e3 - e5 + e35 - a35 + p35,
     1 - e4 - e5 + e45 - a45 + p45,
     1 - e1 - e2 - e3 + e12 + e13 + e23 - e123 - a123 + p123,
     1 - e1 - e2 - e4 + e12 + e14 + e24 - e124 - a124 + p124,
     1 - e1 - e2 - e5 + e12 + e15 + e25 - e125 - a125 + p125,
     1 - e1 - e3 - e4 + e13 + e14 + e34 - e134 - a134 + p134,
     1 - e1 - e3 - e5 + e13 + e15 + e35 - e135 - a135 + p135,
     1 - e1 - e4 - e5 + e14 + e15 + e45 - e145 - a145 + p145,
     1 - e2 - e3 - e4 + e23 + e24 + e34 - e234 - a234 + p234,
     1 - e2 - e3 - e5 + e23 + e25 + e35 - e235 - a235 + p235,
     1 - e2 - e4 - e5 + e24 + e25 + e45 - e245 - a245 + p245,
     1 - e3 - e4 - e5 + e34 + e35 + e45 - e345 - a345 + p345,
     1 - e1 - e2 - e3 - e4 + e12 + e13 + e14 + e23 + e24 + e34 - e123 - e124 - e134 - e234 + e1234 - a1234 + p1234,
     1 - e1 - e2 - e3 - e5 + e12 + e13 + e15 + e23 + e25 + e35 - e123 - e125 - e135 - e235 + e1235 - a1235 + p1235,
     1 - e1 - e2 - e4 - e5 + e12 + e14 + e15 + e24 + e25 + e45 - e124 - e125 - e145 - e245 + e1245 - a1245 + p1245,
     1 - e1 - e3 - e4 - e5 + e13 + e14 + e15 + e34 + e35 + e45 - e134 - e135 - e145 - e345 + e1345 - a1345 + p1345,
     1 - e2 - e3 - e4 - e5 + e23 + e24 + e25 + e34 + e35 + e45 - e234 - e235 - e245 - e345 + e2345 - a2345 + p2345,
     1 - e1 - e2 - e3 - e4 - e5 + e12 + e13 + e14 + e15 + e23 + e24 + e25 + e34 + e35 + e45 - e123 - e124 - e125 - e134 - e135 - e145 - e234 - e235 - e245 - e345 + e1234 + e1235 + e1245 + e1345 + e2345 - e12345 - a12345 + p12345];
 
 c = [e12 - min([e1,e2]),
      e13 - min([e1,e3]),
      e14 - min([e1,e4]),
      e15 - min([e1,e5]),
      e23 - min([e2,e3]),
      e24 - min([e2,e4]),
      e25 - min([e2,e5]),
      e34 - min([e3,e4]),
      e35 - min([e3,e5]),
      e45 - min([e4,e5]),
      e123 - min([e12,e13,e23]),
      e124 - min([e12,e14,e24]),
      e125 - min([e12,e15,e25]),
      e134 - min([e13,e14,e34]),
      e135 - min([e13,e15,e35]),
      e145 - min([e14,e15,e45]),
      e234 - min([e23,e24,e34]),
      e235 - min([e23,e25,e35]),
      e245 - min([e24,e25,e45]),
      e345 - min([e34,e35,e45]),
      e1234 - min([e123,e124,e134,e234]),
      e1235 - min([e123,e125,e135,e235]),
      e1245 - min([e124,e125,e145,e245]),
      e1345 - min([e134,e135,e145,e345]),
      e2345 - min([e234,e235,e245,e345]),
      min([e1,e2,e3,e4,e5]) - .5,
      e12345 - min([e1234,e1235,e1245,e1345,e2345]),
      p12 - e12,
      p13 - e13,
      p14 - e14,
      p15 - e15,
      p23 - e23,
      p24 - e24,
      p25 - e25,
      p34 - e34,
      p35 - e35,
      p45 - e45,
      p123 - e123,
      p124 - e124,
      p125 - e125,
      p134 - e134,
      p135 - e135,
      p145 - e145,
      p234 - e234,
      p235 - e235,
      p245 - e245,
      p345 - e345,
      p1234 - e1234,
      p1235 - e1235,
      p1245 - e1245,
      p1345 - e1345,
      p2345 - e2345,
      p12345 - e12345];
  
 


end