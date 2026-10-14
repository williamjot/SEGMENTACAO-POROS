import unittest

import numpy as np

import poros


class MetricasDePorosTest(unittest.TestCase):
    def test_franja_usa_oito_vizinhos(self):
        nucleo = np.zeros((4, 4), dtype=bool)
        azul = np.zeros_like(nucleo)
        nucleo[1, 1] = True
        azul[2, 2] = True

        self.assertTrue(poros.incluir_franja(nucleo, azul)[2, 2])

    def test_area_minima_e_convertida_para_a_escala(self):
        mascara = np.zeros((8, 8), dtype=bool)
        mascara[1, 1:4] = True
        mascara[4, 1:5] = True

        filtrada = poros.remover_pequenos(mascara, area_minima_px=50, escala=0.25)

        self.assertFalse(filtrada[1, 1])
        self.assertTrue(filtrada[4, 1])

    def test_objeto_de_borda_nao_entra_no_resumo_de_tamanho(self):
        mascara = np.zeros((20, 20), dtype=bool)
        mascara[0:4, 0:4] = True
        mascara[8:13, 8:13] = True

        propriedades = poros.propriedades_poros(mascara, escala=0.5)
        resumo = poros.resumir_poros(mascara, propriedades, escala=0.5)

        self.assertEqual(resumo["n_poros"], 2)
        self.assertEqual(resumo["n_poros_inteiros"], 1)
        self.assertAlmostEqual(resumo["area_mediana_px2_original"], 100.0)

    def test_calibracao_fisica(self):
        mascara = np.zeros((12, 12), dtype=bool)
        mascara[3:7, 3:7] = True

        propriedades = poros.propriedades_poros(
            mascara, escala=1.0, micrometros_por_pixel=0.5
        )
        resumo = poros.resumir_poros(
            mascara,
            propriedades,
            escala=1.0,
            micrometros_por_pixel=0.5,
        )

        self.assertAlmostEqual(propriedades[0]["area_um2"], 4.0)
        self.assertIn("densidade_poros_por_mm2", resumo)

    def test_descritores_basicos_sao_consistentes(self):
        mascara = np.zeros((30, 30), dtype=bool)
        mascara[8:18, 6:21] = True
        mascara[12:16, 16:21] = False

        poro = poros.propriedades_poros(mascara)[0]

        self.assertGreater(poro["razao_aspecto"], 1.0)
        self.assertGreaterEqual(poro["rugosidade"], 1.0)
        self.assertGreater(poro["circularidade"], 0.0)
        self.assertLessEqual(poro["solidez"], 1.0)


if __name__ == "__main__":
    unittest.main()
