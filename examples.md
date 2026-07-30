# Examples

Hard sentence in, simpler sentence out. Nothing here is edited.

### 1

**Before**

> Notwithstanding considerable methodological heterogeneity, the meta-analysis demonstrated a statistically significant attenuation of symptom severity.

**After**

> Despite significant differences in research methods, the meta-analysis found a significant reduction in symptom severity.

*reading grade 35 to 19*

### 2

**Before**

> Inflationary monetary policy, when pursued absent corresponding productivity gains, precipitates a deterioration in aggregate purchasing power.

**After**

> Monetary policy with high inflation can lead to a loss of purchasing power, especially when there is no increase in productivity.

*reading grade 25 to 14*

### 3

**Before**

> The judiciary's interpretive latitude in constitutional adjudication remains contingent upon prevailing doctrinal commitments regarding originalism.

**After**

> The judiciary's power to interpret the constitution is still limited by the prevailing views on originalism in constitutional law.

*reading grade 26 to 16*

### 4

**Before**

> Epigenetic modification encompasses heritable alterations in gene expression unaccompanied by changes to the underlying nucleotide sequence.

**After**

> Epigenetic modifications involve changes to gene expression that are passed down through generations, but do not alter the DNA sequence itself.

*reading grade 25 to 16*

### 5

**Before**

> The mitochondrion is a double-membrane-bound organelle whose principal function is the generation of adenosine triphosphate via oxidative phosphorylation.

**After**

> The mitochondrion is a specialized organelle with two layers of membranes. Its main job is to produce energy in the form of adenosine triphosphate (ATP) through a process called oxidative phosphorylation.

*reading grade 20 to 12*

### 6

**Before**

> The most famous example is the lineage of Dalai Lamas; the current Dalai Lama, Tenzin Gyatso, is held to be the fourteenth mindstream emanation, the first being Gendun Drup (1391 – 1474).

**After**

> The most well-known example is the lineage of Dalai Lamas. The current Dalai Lama, Tenzin Gyatso, is considered the 14th incarnation of the Dalma Lama. The first Dalma Lama was Gendun Drup, born in 1391 and died in 1474.

*reading grade 16 to 8*

### 7

**Before**

> The Book of Nehemiah is a book of the Hebrew Bible, historically regarded as a continuation of the Book of Ezra, and is sometimes called the second book of Ezra.

**After**

> The Book of Nehemiah is a book in the Hebrew Bible, considered a continuation of the Book of Ezra. It is sometimes referred to as the second book of Ezra.

*reading grade 15 to 8*

### 8

**Before**

> Greenland first entertained the idea of a flag of its own in 1973 when five Greenlanders proposed a green, white and blue flag.

**After**

> The concept of a Greenland flag was first considered in 1973. Five Greenlanders proposed a flag with a green, white, and blue design.

*reading grade 9 to 5*

### 9

**Before**

> Although these parties may describe themselves as "libertarian," their ideologies differ considerably and not all of them espouse all elements of the libertarian agenda.

**After**

> While some parties claim to be "libertarian," their ideologies are distinct and not all of them support all aspects of the libertarian agenda.

*reading grade 18 to 14*

### 10

**Before**

> The Feast of the Sacred Heart has been a Solemnity in the Roman Catholic liturgical calendar since 1856, and is celebrated 19 days after Pentecost.

**After**

> The Feast of the Sacred Heart is a solemn Catholic holiday in the Roman Catholic liturgical calendar, dating back to 1856. It is celebrated 19 days after the holiday of Pentecost.

*reading grade 14 to 10*

---

## Turned up too far

The strength is a dial. Past about 1.2 the output starts repeating and stops being usable. `simplify.py` detects this and backs the strength off automatically, so you should not see it, but this is what it looks like.

**Before**

> Photosynthesis constitutes the biochemical process whereby chlorophyllous organisms transduce electromagnetic radiation into chemical potential energy.

**After, strength 2.0**

> Photosynthesis is a biochemical process that helps plants and green plants make food. They use sunlight helps plants make food. They use sunlight makes food. They use sunlight makes food. 
- They have green things help plants make food. 
- They have green things help plants make food. 
- They have green things help plants make food. 
- They have green things help plants make food.

**Before**

> Photosynthesis constitutes the biochemical process whereby chlorophyllous organisms transduce electromagnetic radiation into chemical potential energy.

**After, strength 2.5**

> Photosynthesis is a biochemical process that helps plants and green plants makes a tiny thing is a tiny thing is a tiny thing is a tiny thing is a tiny thing is a big thing. 
- Chlorgy is a big thing. 
- Chlorgy is a big thing. 
- Chlorgy is a big thing. 
- Chlorgy is a big thing. 
- Chlorgy is a big

---

Model `Llama-3.2-3B-Instruct`, structural steering at layer 15, strength 0.75 to 1.0.

Reproduce any of these:

```bash
python src/simplify.py --text "<sentence>" --structural 0.75 --structural-layer 15
```
